import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pair_index = 10

base_dir = Path("models_all")
config_files = list(base_dir.glob("*/*/config.json"))

resnet_models = []

print("🔍 Scanning for all ResNet50 models...")

for cfg_path in tqdm.tqdm(config_files):
    try:
        with open(cfg_path, "r") as f:
            data = json.load(f)
        state = data.get("py/state", {})
        arch = state.get("model_architecture", None)
        poisoned = state.get("poisoned", None)
        num_classes = state.get("number_classes", None)
        if arch == "classification:resnet50" and poisoned is not None and num_classes is not None:
            resnet_models.append({
                "path": cfg_path.parent,
                "poisoned": poisoned,
                "num_classes": num_classes
            })
    except Exception as e:
        print(f"⚠️ Error reading {cfg_path}: {e}")

pairs = []
used_poisoned = set()

clean_models = [m for m in resnet_models if not m["poisoned"]]
poisoned_models = [m for m in resnet_models if m["poisoned"]]

for clean_m in clean_models:
    for poisoned_m in poisoned_models:
        if poisoned_m["path"] not in used_poisoned and clean_m["num_classes"] == poisoned_m["num_classes"]:
            pairs.append((clean_m, poisoned_m))
            used_poisoned.add(poisoned_m["path"])
            break

if len(pairs) < pair_index:
    print(f"\n❌ Only {len(pairs)} valid pairs found. You requested pair #{pair_index}.")
    exit()

selected_pair = pairs[pair_index - 1]
clean_path = selected_pair[0]["path"]
poisoned_path = selected_pair[1]["path"]

print(f"\n✅ Pair #{pair_index} Found (Same num_classes):")
print(f"Clean: {clean_path}")
print(f"Poisoned: {poisoned_path}")
print(f"Number of classes: {selected_pair[0]['num_classes']}")

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

# مسیر ذخیره و suffix مخصوص pair
suffix = f"_pair{pair_index}"
save_dir = Path(f"bn_conv_hist_plots_pair{pair_index}")
os.makedirs(save_dir, exist_ok=True)

def save_hist_per_layer(clean_dict, backdoor_dict, kind, suffix):
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
        filename = f"{kind}_{layer}{suffix}.png"
        plt.savefig(save_dir / filename)
        plt.close()
        print(f"📊 Saved plot: {save_dir / filename}")

save_hist_per_layer(clean_bn_bias, backdoor_bn_bias, "bn_bias", suffix)
save_hist_per_layer(conv_weights_clean, conv_weights_backdoor, "conv_weight", suffix)

def print_stats(label, values, layer):
    print(f"{label} ({layer}):")
    print(f"  Count: {len(values)}")
    print(f"  Std Dev: {np.std(values):.4f}")
    print(f"  Min: {np.min(values):.4f}")
    print(f"  Max: {np.max(values):.4f}\n")

print(f"\n--- 📈 Statistics for Pair #{pair_index} (Layer3 & Layer4, Same num_classes) ---\n")
for layer in target_layers:
    print_stats("Clean Model BN bias", clean_bn_bias[layer], layer)
    print_stats("Backdoored Model BN bias", backdoor_bn_bias[layer], layer)
    print_stats("Clean Model Conv weights", conv_weights_clean[layer], layer)
    print_stats("Backdoored Model Conv weights", conv_weights_backdoor[layer], layer)

print(f"\n✅ All done! Plots are saved in: {save_dir.resolve()}")
