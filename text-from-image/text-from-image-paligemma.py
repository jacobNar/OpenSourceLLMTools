from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.image_utils import load_image
from PIL import Image
import requests
import torch

model_id = "google/paligemma-3b-mix-224"

url = "./processed-images/1.png"
image = load_image(url)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Leaving the prompt blank for pre-trained models
prompt = "<image><bos>the screenshot is an image of a betting lines tab in the order of spread, money and total. What is the spread betting odds for the first row?"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=1000, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)