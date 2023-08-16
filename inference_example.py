from transformers import AutoProcessor, AutoTokenizer
import torch
from PIL import Image

from modelingYING import VLMForConditionalGeneration


# set device
device="cuda:0"

# set prompt template
prompt_template = """
<human>:
{instruction}
{input}
<bot>:
"""

# load processor and tokenizer
processor = AutoProcessor.from_pretrained("MMInstruction/YingVLM")
tokenizer = AutoTokenizer.from_pretrained("MMInstruction/YingVLM") # ziya is not available right now 


# load model
model = VLMForConditionalGeneration.from_pretrained("MMInstruction/YingVLM")
model.to(device,dtype=torch.float16)


# prepare input
image = Image.open("./imgs/night_house.jpeg")
instruction = "Scrutinize the given image and answer the connected question."
input = "What is the color of the couch?"
prompt = prompt_template.format(instruction=instruction, input=input)


# inference
inputs = processor(images=image,  return_tensors="pt").to(device, torch.float16)
text_inputs = tokenizer(prompt, return_tensors="pt")
inputs.update(text_inputs)



generated_ids = model.generate(**{k: v.to(device) for k, v in inputs.items()}, img_num=1, max_new_tokens=128, do_sample=False)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split("\n")[0] # \n is the end token

print(generated_text)
# The couch in the living room is green.

