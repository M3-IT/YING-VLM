from transformers import Blip2ForConditionalGeneration
import torch

class VLMForConditionalGeneration(Blip2ForConditionalGeneration):
    main_input_name = "pixel_values"
    def __init__(self, config):
        super().__init__(config)
        if 'llama' in self.config.text_config.architectures[0].lower():
            print("LLama LM Detect, setting up prefix and suffix tokens")
            self.prefix_ids = torch.tensor([[  529, 25518, 29958]], dtype=torch.long)
            self.suffix_ids = torch.tensor([[ 1533, 25518, 29958]], dtype=torch.long)
        else:
            print("Asumming the lm is neox, setting up prefix and suffix ids")
            self.prefix_ids = torch.tensor([[  29, 40148, 31]], dtype=torch.long)
            self.suffix_ids = torch.tensor([[ 870, 40148, 31]], dtype=torch.long)

    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        input_ids= None,
        attention_mask=None,
        img_num=1,
        **generate_kwargs,
    ) -> torch.LongTensor:

        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.language_projection(query_output)

        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        if self.prefix_ids is not None and self.suffix_ids is not None: # add image prefix and suffix 
            if img_num == 1: 
                prefix_ids = self.prefix_ids.repeat(input_ids.size(0), 1).to(language_model_inputs.device) # bsz, id_len
                suffix_ids = self.suffix_ids.repeat(input_ids.size(0), 1).to(language_model_inputs.device) # bsz, id_len
                prefix_embeds = self.language_model.get_input_embeddings()(prefix_ids) # bsz, id_len, hidden 
                suffix_embeds = self.language_model.get_input_embeddings()(suffix_ids) # bsz, id_len, hidden  
                language_model_inputs = torch.cat( [prefix_embeds, language_model_inputs, suffix_embeds], dim=1)
            else: # multiple images 
                # for a batch, the images pixel values are flattened   
                # e.g.,  [bsz, num_images, channel, height, width] -> [bsz * num_images, c, h, w]
                prefix_ids = self.prefix_ids.repeat(input_ids.size(0), 1).to(language_model_inputs.device) # bsz, id_len
                suffix_ids = self.suffix_ids.repeat(input_ids.size(0), 1).to(language_model_inputs.device) # bsz, id_len
                prefix_embeds = self.language_model.get_input_embeddings()(prefix_ids) # bsz, id_len, hidden 
                suffix_embeds = self.language_model.get_input_embeddings()(suffix_ids) # bsz, id_len, hidden  
                _, number_query_tokens, hidden_dim = language_model_inputs.size()
                bsz = inputs_embeds.size(0)
                split_visual_tokens = language_model_inputs.unsqueeze(0).view(-1, img_num, number_query_tokens, hidden_dim)  # bsz, img_num, num_query_token, hidden size
                split_visual_tokens_with_tokens = []
                for i in range(img_num): # for evey query tokens of each image 
                    img_embeds = split_visual_tokens[:, i, :, :] # bsz, number_query_tokens, hidden 
                    img_embeds_with_tokens = torch.cat([prefix_embeds, img_embeds, suffix_embeds], dim=1) # bsz,  2 * id_len + number_query_tokens, hidden 
                    split_visual_tokens_with_tokens.append(img_embeds_with_tokens)  # 
                combined_visual_tokens = torch.stack(split_visual_tokens_with_tokens, dim=1)
                language_model_inputs = combined_visual_tokens.reshape( bsz , -1 , hidden_dim) # bsz, img_num * (2 * id_len + num_query_tokens), hidden
                assert language_model_inputs.size() == (bsz, img_num * ( prefix_embeds.size(1) + number_query_tokens + suffix_embeds.size(1)), hidden_dim)



        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)


        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs
    