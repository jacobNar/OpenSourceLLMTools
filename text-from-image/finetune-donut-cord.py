import torch
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel, DonutProcessor, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import json
from huggingface_hub import login
from datasets import load_dataset
import os
from PIL import Image
from transformers import default_data_collator
print(torch.__version__)
print(torch.cuda.memory_allocated())


# Load model and processor
config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
config.decoder_start_token_id = 101
config.pad_token_id = 0

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base", padding=False)
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.cuda.memory_summary(device=device)
torch.cuda.empty_cache()

model.to(device)

def preprocess_function(example):
    image = example["image"]
    json_file = f"./json/{os.path.basename(image.filename).split('.')[0]}.json"
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    # Load the text data from the JSON file
    with open(json_file, "r") as f:
        json_data = json.load(f)
        text = json.dumps(json_data.get("games", ""))

    print(text)
    # Combine the processed image and text data for  model input
    encoding = processor(images=image, text=text, return_tensors="pt", padding="max_length", truncation=True, max_length=1204)
    
    return {
        "pixel_values": encoding["pixel_values"].squeeze(0),
        "labels": encoding["labels"].squeeze(0),
    }


# Load data from images and json folders
dataset = load_dataset("imagefolder", data_dir="processed-images")
print(dataset)
dataset = dataset.map(preprocess_function, batched=False)

# Fine-tune the model
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    fp16=True,
    num_train_epochs=1,
    save_strategy="epoch",
    save_total_limit=1,
    predict_with_generate=True,
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=default_data_collator,
    train_dataset=dataset["train"],
)

trainer.train()

# Push the model to Hugging Face
# login()  # Log in to your Hugging Face account
model.push_to_hub("jacobNar/donut-base-finetuned-cord-v2-sports-betting-tables")
processor.push_to_hub("jacobNar/donut-base-finetuned-cord-v2-sports-betting-tables")