import os
import time
from huggingface_hub import InferenceClient

from getpass import getpass

API_KEY = input("Enter HuggingFace API Key: ")
os.environ['HF_TOKEN'] = API_KEY # Set the API key as an environment variable for authentication

# Initialize client with API key for models requiring authentication
client_with_auth = InferenceClient(api_key=os.environ["HF_TOKEN"])

# Initialize client without API key for publicly available models
client = InferenceClient()

# Function to query Llama-3.1-8B-Instruct model, which requires authentication
def query_llama(prompt, system_message=None):
    """
    Query Llama-3.1-8B-Instruct model
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    completion = client_with_auth.chat.completions.create( # Use the authenticated client for Llama model
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        max_tokens=1000
    )
    return completion.choices[0].message.content


# Note: DeepSeek-V3-0324 is a publicly available model, so we can use the client without authentication
def query_deepseek(prompt, system_message=None):
    """
    Query DeepSeek-V3-0324 model
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    completion = client.chat.completions.create( # Use the unauthenticated client for DeepSeek model
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=messages,
        max_tokens=1000
    )
    return completion.choices[0].message.content

prompt = """Can you explain what is quantum computing to a 5th grader?"""
print(prompt)

# Query Llama model
print("=" * 50)
print("LLAMA-3.1-8B-INSTRUCT RESPONSE:")
print("=" * 50)
start_llama = time.perf_counter()
response_llama = query_llama(prompt)
end_llama = time.perf_counter()
print(response_llama)


# Query DeepSeek model

print("=" * 50)
print("DEEPSEEK-V3 RESPONSE:")
print("=" * 50)
start_deepseek = time.perf_counter()
response_deepseek = query_deepseek(prompt)
end_deepseek = time.perf_counter()
print(response_deepseek)
print(f"\n[Response Time for deepseek: {end_deepseek - start_deepseek:.2f} seconds]")
print(f"\n[Response Time for llama: {end_llama - start_llama:.2f} seconds]")



