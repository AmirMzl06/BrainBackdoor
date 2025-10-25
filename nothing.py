import os
import sys
import json
import shutil
import subprocess
from pathlib import Path


PROMPT = "Hi how are you my friend?"

def run_cmd(cmd):
    print("▶️  Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def ensure_packages(packages):
    import importlib
    for pkg in packages:
        name = pkg.split("==")[0].split(">=")[0].split()[0]
        try:
            importlib.import_module(name)
        except Exception:
            run_cmd([sys.executable, "-m", "pip", "install", pkg])

def clone_repo_if_needed(repo_url, dest):
    if dest.exists():
        print(f"Repository already cloned at {dest}")
    else:
        run_cmd(["git", "clone", repo_url, str(dest)])

def download_index_and_shards(repo_id, out_dir):
    from huggingface_hub import hf_hub_download
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🔎 Downloading model.safetensors.index.json from HF...")

    index_filenames = ["model.safetensors.index.json", "pytorch_model.safetensors.index.json"]
    index_path = None
    for fname in index_filenames:
        try:
            local = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="model")
            index_path = Path(local)
            print("index downloaded:", index_path)
            break
        except Exception:
            pass

    if index_path is None:
        raise RuntimeError("نتونستم فایل index.json رو از Hugging Face پیدا کنم. مطمئن شو repo_id درست باشه.")

    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    if "weight_map" in idx:
        mapping = idx["weight_map"]
    elif "weights" in idx:
        mapping = idx["weights"]
    else:
        raise RuntimeError("فرمت index.json ناشناخته؛ کلید weight_map یا weights یافت نشد.")

    shard_files = sorted(set(mapping.values()))
    print("Found shard files:", shard_files)

    local_shards = []
    for shard in shard_files:
        print("⬇️ Download shard:", shard)
        local = hf_hub_download(repo_id=repo_id, filename=shard, repo_type="model")
        local_shards.append(local)
    return local_shards

def merge_safetensor_shards(shard_paths):
    from safetensors.torch import load_file as load_safetensors
    merged = {}
    for p in shard_paths:
        print("Loading safetensor shard:", p)
        data = load_safetensors(p)  # returns dict of torch tensors
        for k, v in data.items():
            merged[k] = v
    print("Merged parameters count:", len(merged))
    return merged

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load micro-llama style model from HF and repo")
    parser.add_argument("--repo-url", default="https://github.com/BKHMSI/mixture-of-cognitive-reasoners.git",
                        help="Git repo of the project")
    parser.add_argument("--repo-dir", default="mixture-of-cognitive-reasoners", help="local folder for repo")
    parser.add_argument("--hf-repo-id", default="bkhmsi/micro-llama-1b", help="HuggingFace repo id for weights/tokenizer")
    parser.add_argument("--work-dir", default="model_data", help="where shards/index are downloaded")
    parser.add_argument("--no-install", action="store_true", help="skip automatic pip installs")
    args = parser.parse_args()

    if not args.no_install:
        ensure_packages([
            "safetensors",
            "transformers>=4.30.0",
            "huggingface_hub",
            "pyyaml",
            "torch",
        ])

    repo_path = Path(args.repo_dir).resolve()
    clone_repo_if_needed(args.repo_url, repo_path)
    sys.path.insert(0, str(repo_path))

    shards = download_index_and_shards(args.hf_repo_id, args.work_dir)
    state_dict = merge_safetensor_shards(shards)

    # try import model class from repo
    model = None
    try:
        mod = __import__("models.micro_llama", fromlist=["MicroLlama"])
        MicroLlama = getattr(mod, "MicroLlama", None)
        if MicroLlama is None:
            for n in ("Micro_Llama", "MicroLlamaModel", "MicroLlamaLM"):
                MicroLlama = getattr(mod, n, None)
                if MicroLlama is not None:
                    break
        if MicroLlama is None:
            raise ImportError("کلاسی به نام MicroLlama در models/micro_llama.py پیدا نشد.")
        print("✅ Found MicroLlama class in repo.")
        # try to read config (optional) to get model args
        cfg_path = None
        for candidate in ["configs/config_micro_llama.yml", "configs/config_micro_llama.yaml", "configs/config.yml"]:
            p = repo_path / candidate
            if p.exists():
                cfg_path = p
                break
        model_args = {}
        if cfg_path:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            model_args = cfg.get("model_args") or cfg.get("model") or {}
            if not isinstance(model_args, dict):
                model_args = {}
        model = MicroLlama(**model_args) if model_args else MicroLlama()
    except Exception as e:
        print("⚠️ Couldn't import/instantiate MicroLlama from repo:", e)
        raise

    # load weights
    try:
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("✅ Weights loaded into model.")
    except Exception as e:
        print("⚠️ خطا هنگام load کردن state_dict به مدل:", e)
        raise

    # tokenizer
    from transformers import AutoTokenizer
    tokenizer = None
    try:
        print("Trying to load tokenizer from HF repo:", args.hf_repo_id)
        tokenizer = AutoTokenizer.from_pretrained(args.hf_repo_id, use_fast=False)
        print("✅ Tokenizer loaded.")
    except Exception as e:
        print("⚠️ نتونستم tokenizer را از HF بارگذاری کنم:", e)
        print("→ اگر repo tokenizer جداست باید مسیرش رو دستی بدهی یا فایل tokenizer.model رو داخل work-dir قرار بدی.")
        # continue; but without tokenizer generation is unreliable

    # device
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === این‌جا از PROMPT استفاده می‌کنیم (متن وورودی داخل کد است) ===
    prompt = PROMPT

    if tokenizer is None:
        raise RuntimeError("بدون tokenizer نمی‌توان به‌طور قابل اعتماد inference انجام داد. لطفاً tokenizer را آماده کن.")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if hasattr(model, "generate"):
            print("Using model.generate(...)")
            try:
                out_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            except TypeError:
                out_ids = model.generate(inputs["input_ids"], max_new_tokens=256, do_sample=False)
            if isinstance(out_ids, (list, tuple)):
                out_ids = out_ids[0]
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            print("\n=== GENERATED ===\n")
            print(text)
        else:
            # fallback naive greedy if no generate
            print("مدل متد generate ندارد؛ تلاش برای دریافت logits و decode به‌صورت greedy (ناپایدار).")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = None
                if isinstance(outputs, dict):
                    logits = outputs.get("logits") or outputs.get("predictions")
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                if logits is None:
                    raise RuntimeError("نتوانستم logits را از خروجی مدل بدست آورم. از generate.py ریپو استفاده کن.")
                cur_input = inputs["input_ids"]
                max_new = 200
                for _ in range(max_new):
                    out = model(input_ids=cur_input)
                    logits = out.logits[:, -1, :]
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                    cur_input = torch.cat([cur_input, next_id], dim=1)
                text = tokenizer.decode(cur_input[0], skip_special_tokens=True)
                print("\n=== GENERATED (naive greedy) ===\n")
                print(text)

if __name__ == "__main__":
    sys.argv = ["run_micro_llama.py"]
    main()
