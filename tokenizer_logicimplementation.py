import os
import time
import psutil
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers
#grant model: login with hf token to access private models
def login_with_token(token):
    login(token=token)

token=input("Enter your Hugging Face API token: ")
login_with_token(token)

# Initialize tokenizer and model for RecurrentGemma-2B-IT where dtype is set to float16 for faster (prediction) inference on compatible hardware. 
# The model is loaded onto the GPU for optimal performance.
model_id="google/recurrentgemma-2b-it"
dtype = torch.float16
tokenizer=  AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="cpu")
#above will load the model on CPU, 
#if you have a compatible GPU, you can change device to "cuda" for faster inference.

# Example of applying the chat template to a user query. 
# The apply_chat_template function formats the input according to the expected structure for chat-based models, adding necessary prompts and system messages if required. The resulting prompt is printed to the console.
chat=[
    {"role": "user", "content": "What is the capital of France?"}
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(prompt)
# The formatted prompt is then tokenized and passed through the model to generate a response. 
# The output is decoded back into human-readable text and printed to the console.

inputs=tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

# Performance monitoring: Start
process = psutil.Process(os.getpid())  # Get the current process for monitoring resource usage
start_time = time.perf_counter()
process.cpu_percent(interval=None) # Initialize Process CPU tracking
psutil.cpu_percent(percpu=True)    # Initialize System-wide per-core tracking
mem_before = process.memory_info().rss / (1024 * 1024) # MB

outputs=model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)

duration = time.perf_counter() - start_time
cpu_usage = process.cpu_percent(interval=None)
per_cpu_usage = psutil.cpu_percent(percpu=True)
mem_after = process.memory_info().rss / (1024 * 1024) # MB
# Performance monitoring: End

response=tokenizer.decode(outputs[0])
print(response)

print(f"\n--- Performance Metrics ---")
print(f"Response Time: {duration:.2f} seconds")
print(f"CPU Utilization: {cpu_usage}%")
print(f"Core-wise Utilization:")
for i, usage in enumerate(per_cpu_usage):
    print(f"  Core {i}: {usage}%")
print(f"Memory Used (Delta): {mem_after - mem_before:.2f} MB")
print(f"Total Resident Memory: {mem_after:.2f} MB")

