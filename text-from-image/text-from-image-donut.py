import re

from transformers import DonutProcessor, VisionEncoderDecoderModel,AutoModelForImageTextToText
from datasets import load_dataset
import torch
from PIL import Image, ImageOps, ImageFilter

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
processor.save_pretrained('./results/checkpoint-1')
model = AutoModelForImageTextToText.from_pretrained("naver-clova-ix/donut-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

#Load image from local path
image = Image.open('./processed-images/2.jpg')

# prepare decoder inputs
prompt = "What are the moneylines for each team?"
# task_prompt = f"<s_cord-v2><s_question>{prompt}</s_question>"
task_prompt = "<s_cord_v2>"
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

sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
print(processor.token2json(sequence))