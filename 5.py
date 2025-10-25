import os
import sys
import subprocess
import torch
from transformers import AutoTokenizer

# ================== نصب کتابخانه‌های موردنیاز ==================
for pkg in ["torch", "transformers", "safetensors"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", pkg])

# ================== کلون کردن مخزن ==================
REPO_URL = "https://github.com/BKHMSI/mixture-of-cognitive-reasoners.git"
PROJECT_PATH = os.path.join(os.getcwd(), "mixture-of-cognitive-reasoners")

if not os.path.exists(PROJECT_PATH):
    print("📥 Cloning Mixture-of-Cognitive-Reasoners repository ...")
    subprocess.run(["git", "clone", REPO_URL, PROJECT_PATH])
else:
    print("✅ Repo already exists, skipping clone.")

# ================== اضافه کردن مسیر به sys.path ==================
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

# ================== وارد کردن کلاس‌های مدل ==================
try:
    from mixture_of_cognitive_reasoners.models.micro_llama import MiCRoLlama, MiCRoLlamaConfig
except ModuleNotFoundError:
    print("❌ Couldn't import MiCRoLlama — check repo path!")
    raise

# ================== بارگذاری مدل ==================
MODEL_ID = "bkhmsi/micro-llama-1b"
print("🔹 Loading tokenizer and config ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
config = MiCRoLlamaConfig.from_pretrained(MODEL_ID)

print("🔹 Loading model weights (this may take several minutes) ...")
model = MiCRoLlama.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ================== تست ورودی و تولید خروجی ==================
input_text = "Explain the role of reasoning modules in brain-like language models."
print(f"\n🧠 Input: {input_text}\n")

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("🧩 Model output:\n")
print(generated_text)
