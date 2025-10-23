import importlib
import subprocess
import sys

package_name = "transformers"
spec = importlib.util.find_spec(package_name)

if spec is None:
    print(f"{package_name} not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
else:
    print(f"{package_name} is already installed.")

package_name = "torch"
spec = importlib.util.find_spec(package_name)

if spec is None:
    print(f"{package_name} not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
else:
    print(f"{package_name} is already installed.")


package_name = "accelerate"
spec = importlib.util.find_spec(package_name)

if spec is None:
    print(f"{package_name} not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
else:
    print(f"{package_name} is already installed.")


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
login()

model_name = "bkhmsi/micro-llama-1b"
tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Pad token was not set. Using EOS token as pad token.")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # force_download = True
)

user_message = "hi how are you?"
prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

response_tokens = outputs[0][inputs.input_ids.shape[-1]:]
response = tokenizer.decode(response_tokens, skip_special_tokens=True)

print("\nResponse:")
print(response)
