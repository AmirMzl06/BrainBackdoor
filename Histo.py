import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Step 1: Find the first clean & poisoned ResNet50 models
# ============================================================

base_dir = Path("models_all")
config_files = list(base_dir.glob("*/*/config.json"))

clean_path = None
poisoned_path = None

print("🔍 Scanning for clean & poisoned ResNet50 models...")

for cfg_path in tqdm.tqdm(config_files):
    try:
        with open(cfg_path, "r") as f:
            data = json.load(f)
        state = data.get("py/state", {})
        arch = state.get("model_architecture", None)
        poisoned = state.get("poisoned", None)

        if arch == "classification:resnet50":
            if poisoned and poisoned_path is None:
                poisoned_path = cfg_path.parent
            elif not poisoned and clean_path is None:
                clean_path = cfg_path.parent

        if clean_path and poisoned_path:
            break

    except Exception as e:
        print(f"⚠️ Error reading {cfg_path}: {e}")

if not clean_path or not poisoned_path:
    print("\n❌ Couldn't find both clean and poisoned ResNet50 models.")
    print(f"Clean found: {clean_path}")
    print(f"Poisoned found: {poisoned_path}")
    exit()

print(f"\n✅ Found Clean Model: {clean_path}")
print(f"✅ Found Poisoned Model: {poisoned_path}")

# ============================================================
# Step 2: Load both models
# ============================================================

def load_model(path):
    model_path = path / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found.")
    model = torch.load(model_path, map_location=device)
    if isinstance(model, dict) and "state_dict" in model:
        state_dict = model["state_dict"]
    elif isinstance(model, dict):
        state_dict = model
    else:
        state_dict = model.state_dict()
    return state_dict

CleanModel = load_model(clean_path)
BackdooredModelN = load_model(poisoned_path)

print("✅ Models loaded successfully.\n")

# ============================================================
# Step 3: Extract BN bias and Conv weights
# ============================================================

target_layers = ["layer3", "layer4"]

def extract_params(state_dict, kind):
    values = []
    for name, param in state_dict.items():
        if any(layer in name for layer in target_layers):
            if kind == "bn_bias" and "bn" in name and "bias" in name:
                values.extend(param.cpu().detach().numpy().flatten())
            elif kind == "conv_weight" and "conv" in name and "weight" in name:
                values.extend(param.cpu().detach().numpy().flatten())
    return values

clean_bn_bias_values = extract_params(CleanModel, "bn_bias")
backdoor_bn_bias_values = extract_params(BackdooredModelN, "bn_bias")

conv_weights_clean = extract_params(CleanModel, "conv_weight")
conv_weights_backdoor = extract_params(BackdooredModelN, "conv_weight")

# ============================================================
# Step 4: Plot & Save
# ============================================================

save_dir = Path("bn_conv_hist_plots")
os.makedirs(save_dir, exist_ok=True)

def save_hist(clean_values, backdoor_values, title, filename, label_clean, label_backdoor):
    plt.figure(figsize=(14, 7))
    sns.histplot(clean_values, color='blue', label=label_clean,
                 kde=True, stat='density', alpha=0.6, bins=50)
    sns.histplot(backdoor_values, color='red', label=label_backdoor,
                 kde=True, stat='density', alpha=0.6, bins=50)

    plt.yscale('log')
    plt.title(title, fontsize=16)
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Density (Log Scale)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_dir / filename)
    plt.close()
    print(f"📊 Saved plot: {save_dir / filename}")

# BN Bias plot
save_hist(
    clean_bn_bias_values,
    backdoor_bn_bias_values,
    "Log-Scale Histogram of BN Bias Values (Layer3 & Layer4)",
    "bn_bias_hist.png",
    "Clean Model bn.bias",
    "Backdoored Model bn.bias"
)

# Conv Weights plot
save_hist(
    conv_weights_clean,
    conv_weights_backdoor,
    "Log-Scale Histogram of Conv2d Weights (Layer3 & Layer4)",
    "conv_weight_hist.png",
    "Clean Model Conv Weights",
    "Backdoored Model Conv Weights"
)

# ============================================================
# Step 5: Print basic stats
# ============================================================

def print_stats(label, values):
    print(f"{label}:")
    print(f"  Count: {len(values)}")
    print(f"  Std Dev: {np.std(values):.4f}")
    print(f"  Min: {np.min(values):.4f}")
    print(f"  Max: {np.max(values):.4f}\n")

print("\n--- 📈 Statistics for BN Bias Values (Layer3 & Layer4) ---")
print_stats("Clean Model", clean_bn_bias_values)
print_stats("Backdoored Model", backdoor_bn_bias_values)

print("--- 📈 Statistics for Conv Weights (Layer3 & Layer4) ---")
print_stats("Clean Model", conv_weights_clean)
print_stats("Backdoored Model", conv_weights_backdoor)

print("\n✅ All done! Plots are saved in:", save_dir.resolve())
