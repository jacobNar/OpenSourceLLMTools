import torch
from transformers import pipeline
print(torch.cuda.is_available())  
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",  # replace with "mps" to run on a Mac device
)

print("enter prompt:")
prompt = input()

messages = [
    {"role": "user", "content": prompt},
]

outputs = pipe(messages, max_new_tokens=1024)
assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
print(assistant_response)