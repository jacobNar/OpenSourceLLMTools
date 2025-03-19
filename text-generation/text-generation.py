from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

prompt = 'how many tokens can you accept as input?'
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

tokenizer.pad_token = tokenizer.eos_token
# Generate
generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=1024)
result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(result)