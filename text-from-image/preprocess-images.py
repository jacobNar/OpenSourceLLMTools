import os
from PIL import Image, ImageOps, ImageFilter

# Input and output directories
input_dir = "./images"
output_dir = "./processed-images"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Scale factor
scale_factor = 2

# Loop through all files in the input directory
for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)
    
    # Ensure it's a file and has a valid image extension
    if os.path.isfile(input_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
        try:
            # Open and process the image
            image = Image.open(input_path).convert("RGB")
            image = ImageOps.autocontrast(image)
            
            # Scale image
            width, height = image.size
            image = image.resize((int(width * scale_factor), int(height * scale_factor)), Image.Resampling.LANCZOS)
            
            # Remove speckles
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Sharpen the image
            image = image.filter(ImageFilter.SHARPEN)
            
            # Save the processed image
            output_path = os.path.join(output_dir, file_name)
            image.save(output_path)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")
