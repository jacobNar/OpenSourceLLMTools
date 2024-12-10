import re

from transformers import DonutProcessor, VisionEncoderDecoderModel,AutoModelForImageTextToText
from datasets import load_dataset
import torch
from PIL import Image, ImageOps, ImageFilter

processor = DonutProcessor.from_pretrained("jacobNar/donut-base-finetuned-cord-v2-sports-betting-tables")
processor.save_pretrained('./results/checkpoint-1')
model = AutoModelForImageTextToText.from_pretrained("jacobNar/donut-base-finetuned-cord-v2-sports-betting-tables")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# load document image
# dataset = load_dataset("hf-internal-testing/example-documents", split="test")
# image = dataset[2]["image"]

#Load image from local path
image = Image.open('./images/1.png').convert("RGB")
image = ImageOps.autocontrast(image)

#scale image
scale_factor = 2
width, height = image.size
image = image.resize((int(width * scale_factor), int(height * scale_factor)),  Image.Resampling.LANCZOS)

#remove speckles 
image = image.filter(ImageFilter.MedianFilter(size=3))

image = image.filter(ImageFilter.SHARPEN)


# prepare decoder inputs
prompt = "pull the odds data out of the image like moneyline, spread, and totals."
task_prompt = f"<s_cord-v2><s_question>{prompt}</s_question>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

pixel_values = processor(image, return_tensors="pt").pixel_values

outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

print(outputs)

sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
print(processor.token2json(sequence))