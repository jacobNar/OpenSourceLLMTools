import os
from pathlib import Path
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from PIL import Image
import torch

# 1. Load Latest Model optimized for 16GB VRAM
# Using FLUX.1-dev for state-of-the-art generation quality.
# Note: FLUX.1-dev is a gated model. Ensure you are logged in via `huggingface-cli login`
# or have your HF_TOKEN environment variable set.
# If you prefer the open (Apache 2.0) version, change "black-forest-labs/FLUX.1-dev"
# to "black-forest-labs/FLUX.1-schnell".

print("Loading FLUX.1-dev model (this may take a moment)...")

model_id = "black-forest-labs/FLUX.1-dev"

# Load the main Text-to-Image pipeline
pipeline = FluxPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for best FLUX performance (requires Ampere+ GPU)
    # If on older GPU (Pascal/Volta), change to torch.float16
)

# Enable memory optimizations for 16GB VRAM
# Offloads the model to CPU when not actively processing layers
pipeline.enable_model_cpu_offload()
# pipeline.enable_sequential_cpu_offload() # Uncomment if you still hit OOM errors


# Create Image-to-Image pipeline sharing the same components to save RAM
edit_pipeline = FluxImg2ImgPipeline.from_pipe(pipeline)
# Note: we don't need to call enable_model_cpu_offload on edit_pipeline as it shares the model


def makeimage(prompt):
    """Generate a new image from a text prompt using FLUX.1"""
    print(f"Generating image with FLUX.1-dev: '{prompt}'...")
    
    # FLUX typically works best at 1024x1024 or other megapixel resolutions
    image = pipeline(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5, # FLUX.1-dev works well with lower guidance (3.5 is default)
        num_inference_steps=28, # FLUX.1-dev is efficient, 28-30 steps is standard good quality
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0) # optional for reproducibility
    ).images[0]

    output_dir = Path("./generated-images")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_count = len(list(output_dir.glob("generated-img*.jpg")))
    output_path = f"./generated-images/generated-img{image_count + 1}.jpg"
    image.save(output_path)
    print(f"Image saved to {output_path}")
    return image


def edit_image(image_path, target_prompt):
    """Edit an existing image using FLUX.1 Image-to-Image

    Args:
        image_path: Path to the image to edit
        target_prompt: Text DESCRIPTION of the target image (e.g. "a snowy winter landscape")
                       Note: Unlike InstructPix2Pix, this is NOT an instruction like "make it winter".
    """
    print(f"Editing image with target description: '{target_prompt}'...")
    
    # Resize/Load image
    init_image = Image.open(image_path).convert("RGB")
    # Resize to 1024x1024 for FLUX consistency, or keep aspect ratio if preferred
    init_image = init_image.resize((1024, 1024))

    # FLUX Img2Img editing
    # 'strength' determines how much to change the image. 
    # 0.0 = no change, 1.0 = complete retcon. 0.75-0.85 is usually good for heavy edits.
    edited_image = edit_pipeline(
        prompt=target_prompt,
        image=init_image,
        strength=0.75, 
        guidance_scale=3.5,
        num_inference_steps=28,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]

    output_dir = Path("./generated-images")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_count = len(list(output_dir.glob("edited-img*.jpg")))
    output_path = f"./generated-images/edited-img{image_count + 1}.jpg"
    edited_image.save(output_path)
    print(f"Edited image saved to {output_path}")
    return edited_image


# Main interactive mode
print("FLUX.1 Image Generation & Editing Tool")
print("Commands:")
print("  'gen' or 'g'  - Generate new image")
print("  'edit' or 'e' - Edit existing image (via Img2Img)")
print("  'quit' or 'q' - Exit program")
print()

while True:
    choice = input("Enter command: ").strip().lower()

    if choice in ["quit", "q"]:
        print("Exiting...")
        break
    elif choice in ["gen", "g", "1"]:
        prompt = input("Enter your prompt (or 'back' to return): ").strip()
        if prompt.lower() != "back":
            makeimage(prompt)
        print()
    elif choice in ["edit", "e", "2"]:
        image_path = input(
            "Enter path to image to edit (or 'back' to return): ").strip()
        if image_path.lower() != "back":
            print("NOTE: FLUX is an Image-to-Image model, not instruction-based.")
            print("Please describe the FULL target image, not just the change.")
            target_prompt = input(
                "Enter target prompt (e.g. 'a photo of a cat in the snow' NOT 'add snow'): ").strip()
            if target_prompt.lower() != "back":
                edit_image(image_path, target_prompt)
        print()
    else:
        print("Invalid command. Try 'gen', 'edit', or 'quit'.")
        print()
