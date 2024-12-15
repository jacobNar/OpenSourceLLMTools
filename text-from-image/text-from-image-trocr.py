from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline
from PIL import Image

image = Image.open('./images/3.jpg')
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-str')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-str')
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values, max_length=100,num_beams=2)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_text)
