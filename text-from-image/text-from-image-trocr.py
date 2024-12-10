from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

image = Image.open('./images/2.jpg').convert("L")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-str')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-str')
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_text)
