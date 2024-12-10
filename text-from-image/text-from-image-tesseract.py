from pytesseract import image_to_string, pytesseract
from PIL import Image, ImageOps, ImageFilter

image_path = "./processed-images/3.jpg"

# Path to Tesseract executable (only needed on Windows)
pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

image = Image.open(image_path)
image.convert("L")
image = ImageOps.autocontrast(image)

#scale image
scale_factor = 2
width, height = image.size
image = image.resize((int(width * scale_factor), int(height * scale_factor)),  Image.Resampling.LANCZOS)

#remove speckles 
image = image.filter(ImageFilter.MedianFilter(size=3))

image = image.filter(ImageFilter.SHARPEN)

text = image_to_string(image)

print("Extracted Text:")
print(text)
