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
# 2. Prepare Dataset (Images Only)
# data_dir = "/"  # Change to your dataset folder
# dataset = load_dataset("training-images", data_dir=data_dir)

# # 3. Define Fine-Tuning Configuration
# training_config = StableDiffusionTrainingConfig(
#     num_train_epochs=5,  # Number of fine-tuning epochs
#     per_device_train_batch_size=4,  # Adjust based on your GPU memory (3060 ~12GB VRAM)
#     learning_rate=5e-6,  # Fine-tuning requires a small learning rate
#     output_dir="./fine_tuned_stable_diffusion",  # Directory to save fine-tuned model
#     save_steps=1000,  # Save checkpoint every 1000 steps
# )

# # 4. Initialize Trainer
# trainer = StableDiffusionTrainer(
#     pipeline=pipeline,
#     training_config=training_config,
#     train_dataset=dataset,
# )

# # 5. Start Training
# trainer.train()

# # 6. Save Fine-Tuned Model
# pipeline.save_pretrained("./fine_tuned_stable_diffusion")
