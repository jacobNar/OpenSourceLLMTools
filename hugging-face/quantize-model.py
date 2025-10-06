from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import HfApi, login, create_repo
import torch
import os

HF_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
QUANTIZED_MODEL_ID = "jacobNar/Llama-3.1-3B-Instruct-quantized"
LOCAL_MODEL_DIR = "llama3-1-3-b-quantized"

print("Logging into Hugging Face...")
hf_token = 'YOUR token here'
login(token=hf_token)

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Quantization Configuration optimized for GPU
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    # Changed to float16 for better GPU compatibility
    bnb_4bit_compute_dtype=torch.float16
)

# b. Load and Quantize the Model Locally
print(f"Loading and quantizing model: {HF_MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",  # Automatically handles device placement
    torch_dtype=torch.float16,  # Use float16 for GPU efficiency
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)

# c. Save the Quantized Model to a local directory
print(f"Saving quantized model to {LOCAL_MODEL_DIR}")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
model.save_pretrained(LOCAL_MODEL_DIR)
tokenizer.save_pretrained(LOCAL_MODEL_DIR)

# d. Upload Quantized Model to Hugging Face Hub
print(f"Uploading quantized model to Hugging Face Hub: {QUANTIZED_MODEL_ID}")
api = HfApi()
create_repo(repo_id=QUANTIZED_MODEL_ID, exist_ok=True, token=hf_token)
api.upload_folder(
    folder_path=LOCAL_MODEL_DIR,
    repo_id=QUANTIZED_MODEL_ID,
    repo_type="model",
    token=hf_token
)
