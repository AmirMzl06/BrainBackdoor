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
model_name = "bkhmsi/micro-llama-1b"
from huggingface_hub import login
login()
from transformers import LlamaTokenizer

tokenizer_name = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32,
    force_download = True
)
prompt = "Hi How Are You?"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0]))
