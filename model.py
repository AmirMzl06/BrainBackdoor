from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = "bkhmsi/micro-llama-3b"
from huggingface_hub import login
login("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")
from transformers import LlamaTokenizer

tokenizer_name = "baseten/Meta-Llama-3-tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
prompt = "Hi How Are You?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0]))
