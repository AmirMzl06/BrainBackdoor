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
    """Loads the FC layer weights from a model."""
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
                print(f"⚠️ Warning: FC layer not found in {model_path.parent.name}")
                return None
        
        return fc_weights.detach().cpu().numpy()
    except Exception as e:
        print(f" Error processing model {model_path.parent.name}: {e}")
        return None

def calculate_effective_rank(weights_np):
    """Calculates effective rank using SVD."""
    if weights_np is None:
        return 0.0
    s = np.linalg.svd(weights_np, compute_uv=False)
    tol = np.max(weights_np.shape) * np.finfo(float).eps * np.max(s)
    eff_rank = np.sum(s > tol)
    return eff_rank

# -----------------------------------------------------------------
# --- Plot function ---
# -----------------------------------------------------------------
def plot_global_effective_rank(clean_ranks, poisoned_ranks, save_dir):
    num_clean = len(clean_ranks)
    num_poisoned = len(poisoned_ranks)

    if num_clean == 0 and num_poisoned == 0:
        print("No data to plot.")
        return

    x_clean = np.arange(1, num_clean + 1)
    x_poisoned = np.arange(1, num_poisoned + 1)

    plt.figure(figsize=(15, 8))
    
    if num_clean > 0:
        plt.plot(x_clean, clean_ranks, label=f'Clean Models ({num_clean})', 
                 color='blue', linestyle='--', marker='o', markersize=4, alpha=0.7)
    if num_poisoned > 0:
        plt.plot(x_poisoned, poisoned_ranks, label=f'Poisoned Models ({num_poisoned})', 
                 color='red', linestyle='-', marker='x', markersize=4, alpha=0.7)

    plt.xlabel("Model Index (Sequentially Processed)")
    plt.ylabel("Effective Rank")
    plt.title("Global FC Layer Effective Rank: Clean vs Poisoned Models")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    filename = "GLOBAL_effective_rank_comparison.png"
    save_path = save_dir / filename
    plt.savefig(save_path)
    plt.close()
    
    print(f"\n📈 Global plot saved to:\n{save_path.resolve()}")

# --- Main Processing Function ---
def calculate_and_plot_global_effective_rank():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    main_path = Path('models_all')
    plot_dir = Path('effective_rank_plots')
    plot_dir.mkdir(exist_ok=True)

    if not main_path.exists():
        print(f"Error: '{main_path.resolve()}' not found.")
        return

    config_files = list(main_path.glob('*/*/config.json'))
    clean_ranks = []
    poisoned_ranks = []

    for config_path in tqdm.tqdm(config_files, desc="Processing all models"):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            state = config_data.get('py/state', {})
            is_poisoned = state.get('poisoned', False)

            model_path = config_path.parent / 'model.pt'
            if model_path.exists():
                fc_weights = load_fc_weights(model_path, device)
                eff_rank = calculate_effective_rank(fc_weights)

                if is_poisoned:
                    poisoned_ranks.append(eff_rank)
                else:
                    clean_ranks.append(eff_rank)
        except Exception as e:
            print(f" Error reading {config_path}: {e}")
            continue

    print("\n" + "="*70)
    print("📊 Effective rank calculation finished.")
    print(f"  {len(clean_ranks)} Clean models processed.")
    print(f"  {len(poisoned_ranks)} Poisoned models processed.")
    print("="*70)

    plot_global_effective_rank(clean_ranks, poisoned_ranks, plot_dir)
    print("\n✅ Processing complete.")

if __name__ == "__main__":
    calculate_and_plot_global_effective_rank()
