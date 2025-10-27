import subprocess
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm

# --- Install dependencies ---
try:
    print("Installing/Updating 'timm'...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "timm"])
except Exception as e:
    print(f"Error installing 'timm': {e}")
    sys.exit(1)

# --- Helper Functions ---

def load_fc_weights(model_path, device):
    """
    Loads the model from the given path and extracts the FC layer weights.
    Tries both 'classifier.1.weight' and 'fc.weight'.
    """
    try:
        model_state_dict = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(model_state_dict, dict) and "state_dict" in model_state_dict:
            state_dict = model_state_dict["state_dict"]
        elif isinstance(model_state_dict, dict):
            state_dict = model_state_dict
        else:
            state_dict = model_state_dict.state_dict()

        fc_weights = None
        
        try:
            fc_weights = state_dict['classifier.1.weight']
        except KeyError:
            try:
                fc_weights = state_dict['fc.weight']
            except KeyError:
                print(f"⚠️ Missing FC layer in model {model_path.parent.name}")
                return None
        
        return fc_weights.detach().cpu().numpy()

    except Exception as e:
        print(f"❌ Error processing model {model_path.parent.name}: {e}")
        return None

def calculate_max_z_score(weights_np):
    """
    Calculates the Max Z-Score (Outlier Score) from FC layer weights.
    """
    if weights_np is None or weights_np.size == 0:
        return 0.0

    class_norms = np.linalg.norm(weights_np, axis=1)
    mean_norm = np.mean(class_norms)
    std_norm = np.std(class_norms)

    if std_norm < 1e-9:
        return 0.0

    z_scores = (class_norms - mean_norm) / std_norm
    return float(np.max(z_scores))

# --- Plot Function ---
def plot_global_zscore_graph(clean_zscores, poisoned_zscores, save_dir):
    """
    Plots a line graph comparing the Max Z-Score of all Clean vs. Poisoned models.
    """
    num_clean = len(clean_zscores)
    num_poisoned = len(poisoned_zscores)

    if num_clean == 0 and num_poisoned == 0:
        print("No data to plot.")
        return

    x_clean = np.arange(1, num_clean + 1)
    x_poisoned = np.arange(1, num_poisoned + 1)

    plt.figure(figsize=(15, 8))

    if num_clean > 0:
        plt.plot(
            x_clean, clean_zscores,
            label=f'Clean Models ({num_clean})',
            color='blue', linestyle='--', marker='o', markersize=4, alpha=0.7
        )

    if num_poisoned > 0:
        plt.plot(
            x_poisoned, poisoned_zscores,
            label=f'Poisoned Models ({num_poisoned})',
            color='red', linestyle='-', marker='x', markersize=4, alpha=0.7
        )

    plt.xlabel("Model Index")
    plt.ylabel("Max Z-Score")
    plt.title("Global Max Z-Score Comparison (SVD-based Outlier Metric)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "GLOBAL_max_zscore_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"\n📈 Global Max Z-Score plot saved to:\n{save_path.resolve()}")

# --- Main Function ---
def calculate_and_plot_global_zscores():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶️ Using device: '{device}'")

    main_path = Path("models_all")
    results_dir = Path("svd_zscore_plots")
    results_dir.mkdir(exist_ok=True)

    if not main_path.exists():
        print(f"❌ Directory '{main_path.resolve()}' not found.")
        return

    config_files = list(main_path.glob("*/*/config.json"))

    all_clean_zscores = []
    all_poisoned_zscores = []

    print("🔍 Scanning models...")
    for config_path in tqdm.tqdm(config_files, desc="Processing models"):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            state = config_data.get("py/state", {})
            is_poisoned = state.get("poisoned", False)
            model_path = config_path.parent / "model.pt"

            if not model_path.exists():
                continue

            fc_weights = load_fc_weights(model_path, device)
            if fc_weights is not None:
                zscore = calculate_max_z_score(fc_weights)
                if is_poisoned:
                    all_poisoned_zscores.append(zscore)
                else:
                    all_clean_zscores.append(zscore)

        except Exception as e:
            print(f"⚠️ Error reading {config_path}: {e}")
            continue

    print("\n" + "="*70)
    print("📊 Z-Score Calculation Complete")
    print(f"   {len(all_clean_zscores)} Clean models processed")
    print(f"   {len(all_poisoned_zscores)} Poisoned models processed")
    print("="*70)

    plot_global_zscore_graph(all_clean_zscores, all_poisoned_zscores, results_dir)
    print("\n✅ Processing complete.")

if __name__ == "__main__":
    calculate_and_plot_global_zscores()
