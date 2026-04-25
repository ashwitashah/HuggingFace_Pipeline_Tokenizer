import os
import time
import psutil
from huggingface_hub import login
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

#grant model: login with hf token to access private models
def login_with_token(token):
    login(token=token)

token=input("Enter your Hugging Face API token: ")
login_with_token(token)

# pipeline for text generation using the RecurrentGemma-2B-IT model, which is a chat-based language model.
#  The pipeline abstracts away the complexities of tokenization and model inference, 
# allowing for straightforward generation of responses based on user input messages.
#  The messages are formatted according to the expected structure for chat-based interactions,
#  and the generated response is printed to the console.

pipe = pipeline("text-generation", model="google/recurrentgemma-2b-it")
messages = [
	{"role": "user", "content": "Who are you?"},
]

# Performance monitoring: Pipeline
# Measure the time taken for the pipeline to generate a response, which includes tokenization, model inference, and decoding. This provides insight into the overall performance of the pipeline approach compared to direct model generation.

process = psutil.Process(os.getpid())
start_time_pipe = time.perf_counter()
process.cpu_percent(interval=None)
psutil.cpu_percent(percpu=True)
mem_before_pipe = process.memory_info().rss / (1024 * 1024)

print(pipe(messages))

pipeline_duration = time.perf_counter() - start_time_pipe
pipeline_cpu = process.cpu_percent(interval=None)
pipeline_mem_delta = (process.memory_info().rss / (1024 * 1024)) - mem_before_pipe

print(f"Pipeline Inference Time: {pipeline_duration:.2f}s") 
print("-" * 30)
#CPU and Memory usage for pipeline approach
per_cpu_usage = psutil.cpu_percent(percpu=True)
print(f"CPU Utilization: {pipeline_cpu}%")
print(f"Core-wise Utilization:")
for i, usage in enumerate(per_cpu_usage):
	print(f"  Core {i}: {usage}%")
print(f"Memory Used (Delta): {pipeline_mem_delta:.2f} MB")
print(f"Total Resident Memory: {process.memory_info().rss / (1024 * 1024):.2f} MB")
# Direct generation using the model and tokenizer, 
# which allows for more granular control over the tokenization and generation process. 
# This approach can provide insights into the performance of the model itself, 
# separate from the overhead introduced by the pipeline abstraction. 
# Performance metrics such as response time, CPU utilization, and memory usage are measured to evaluate the efficiency of direct model generation compared to using the pipeline.

process = psutil.Process(os.getpid())
tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-2b-it") # Initialize the tokenizer for the RecurrentGemma-2B-IT model, which is responsible for converting text into token IDs that the model can process. The tokenizer also handles the application of chat templates, ensuring that the input messages are formatted correctly for the model's expectations.
model = AutoModelForCausalLM.from_pretrained("google/recurrentgemma-2b-it")# Load the RecurrentGemma-2B-IT model, which is a causal language model designed for generating text based on input prompts. The model is loaded onto the CPU by default, but can be moved to a compatible GPU for faster inference if available. This setup allows for direct interaction with the model using tokenized inputs and outputs.
messages = [
	{"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template( # Apply the chat template to the input messages, which formats the messages according to the expected structure for chat-based models. This includes adding necessary prompts and system messages if required. The resulting prompt is then tokenized and converted into tensors that can be processed by the model.
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

# Performance monitoring: Direct Generation
start_time_direct = time.perf_counter()
process.cpu_percent(interval=None)
psutil.cpu_percent(percpu=True)
mem_before_direct = process.memory_info().rss / (1024 * 1024)

outputs = model.generate(**inputs, max_new_tokens=40)

direct_duration = time.perf_counter() - start_time_direct
direct_cpu = process.cpu_percent(interval=None)
per_cpu_usage = psutil.cpu_percent(percpu=True)
direct_mem_delta = (process.memory_info().rss / (1024 * 1024)) - mem_before_direct

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

print(f"\n--- Performance Metrics (Direct Generation) ---")
print(f"Response Time for Direct Generation: {direct_duration:.2f} seconds")
print(f"CPU Utilization: {direct_cpu}%")
print(f"Core-wise Utilization:")
for i, usage in enumerate(per_cpu_usage):
	print(f"  Core {i}: {usage}%")
print(f"Memory Used (Delta): {direct_mem_delta:.2f} MB")
print(f"Total Resident Memory: {process.memory_info().rss / (1024 * 1024):.2f} MB")

# Compare response time for both and give insights on which approach is more efficient and by how much.
if pipeline_duration < direct_duration:
	print(f"\nPipeline approach is faster than Direct Generation by {direct_duration - pipeline_duration:.2f} seconds.")	
else:
	print(f"\nDirect Generation is faster than Pipeline approach by {pipeline_duration - direct_duration:.2f} seconds.")	

#Compare each core cpu utilization for both approaches and give insights on which approach is more efficient in terms of CPU usage.
if pipeline_cpu < direct_cpu:
	print(f"\nPipeline approach is more efficient in terms of CPU usage by {direct_cpu - pipeline_cpu:.2f}%.")
else:
	print(f"\nDirect Generation is more efficient in terms of CPU usage by {pipeline_cpu - direct_cpu:.2f}%.")

# Compare memory usage for both approaches and give insights on which approach is more efficient in terms of memory usage.
if pipeline_mem_delta < direct_mem_delta:
	print(f"\nPipeline approach is more efficient in terms of memory usage by {direct_mem_delta - pipeline_mem_delta:.2f} MB.")
else:
	print(f"\nDirect Generation is more efficient in terms of memory usage by {pipeline_mem_delta - direct_mem_delta:.2f} MB.")
