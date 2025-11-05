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
# Step 3: Extract BN bias and Conv weights per layer
# ============================================================

target_layers = ["layer3", "layer4"]

def extract_params_per_layer(state_dict, kind):
    result = {}
    for layer in target_layers:
        values = []
        for name, param in state_dict.items():
            if layer in name:
                if kind == "bn_bias" and "bn" in name and "bias" in name:
                    values.extend(param.cpu().detach().numpy().flatten())
                elif kind == "conv_weight" and "conv" in name and "weight" in name:
                    values.extend(param.cpu().detach().numpy().flatten())
        result[layer] = values
    return result

clean_bn_bias = extract_params_per_layer(CleanModel, "bn_bias")
backdoor_bn_bias = extract_params_per_layer(BackdooredModelN, "bn_bias")

conv_weights_clean = extract_params_per_layer(CleanModel, "conv_weight")
conv_weights_backdoor = extract_params_per_layer(BackdooredModelN, "conv_weight")

# ============================================================
# Step 4: Plot & Save
# ============================================================

save_dir = Path("bn_conv_hist_plots")
os.makedirs(save_dir, exist_ok=True)

def save_hist_per_layer(clean_dict, backdoor_dict, kind):
    for layer in target_layers:
        plt.figure(figsize=(14, 7))
        sns.histplot(clean_dict[layer], color='blue', label=f"Clean {kind} {layer}",
                     kde=True, stat='density', alpha=0.6, bins=50)
        sns.histplot(backdoor_dict[layer], color='red', label=f"Backdoored {kind} {layer}",
                     kde=True, stat='density', alpha=0.6, bins=50)
        plt.yscale('log')
        plt.title(f"Log-Scale Histogram of {kind} Values ({layer})", fontsize=16)
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('Density (Log Scale)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        filename = f"{kind}_{layer}.png"
        plt.savefig(save_dir / filename)
        plt.close()
        print(f"📊 Saved plot: {save_dir / filename}")

# Save BN bias histograms
save_hist_per_layer(clean_bn_bias, backdoor_bn_bias, "bn_bias")
# Save Conv weight histograms
save_hist_per_layer(conv_weights_clean, conv_weights_backdoor, "conv_weight")

# ============================================================
# Step 5: Print basic stats per layer
# ============================================================

def print_stats(label, values, layer):
    print(f"{label} ({layer}):")
    print(f"  Count: {len(values)}")
    print(f"  Std Dev: {np.std(values):.4f}")
    print(f"  Min: {np.min(values):.4f}")
    print(f"  Max: {np.max(values):.4f}\n")

print("\n--- 📈 Statistics per layer ---\n")
for layer in target_layers:
    print_stats("Clean Model BN bias", clean_bn_bias[layer], layer)
    print_stats("Backdoored Model BN bias", backdoor_bn_bias[layer], layer)
    print_stats("Clean Model Conv weights", conv_weights_clean[layer], layer)
    print_stats("Backdoored Model Conv weights", conv_weights_backdoor[layer], layer)

print("\n✅ All done! Plots are saved in:", save_dir.resolve())
