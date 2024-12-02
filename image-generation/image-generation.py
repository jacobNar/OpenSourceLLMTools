import os
from pathlib import Path
from diffusers import StableDiffusionPipeline
from datasets import load_dataset

# 1. Load Pre-Trained Stable Diffusion Model
model_name = "CompVis/stable-diffusion-v1-4"  # Pre-trained Stable Diffusion model
pipeline = StableDiffusionPipeline.from_pretrained(
    model_name,
)

def dummy(images, **kwargs):
    return images, None

pipeline.safety_checker = dummy

def makeimage(prompt):
    image = pipeline(prompt).images[0]
    output_path = "./generated-images/generated-img2.jpg"
    image.save(output_path)
    print(f"Image saved to {output_path}")

print("Enter your prompt: ")
prompt = input()
makeimage(prompt)
