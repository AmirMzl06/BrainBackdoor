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

def calculate_condition_number(weights_np):
    """Calculates condition number of a weight matrix."""
    if weights_np is None:
        return np.nan
    try:
        cond = np.linalg.cond(weights_np)
        return cond
    except np.linalg.LinAlgError:
        return np.inf

# -----------------------------------------------------------------
# --- Plot function ---
# -----------------------------------------------------------------
def plot_global_condition_number(clean_conds, poisoned_conds, save_dir):
    num_clean = len(clean_conds)
    num_poisoned = len(poisoned_conds)

    if num_clean == 0 and num_poisoned == 0:
        print("No data to plot.")
        return

    x_clean = np.arange(1, num_clean + 1)
    x_poisoned = np.arange(1, num_poisoned + 1)

    plt.figure(figsize=(15, 8))
    
    if num_clean > 0:
        plt.plot(x_clean, clean_conds, label=f'Clean Models ({num_clean})', 
                 color='blue', linestyle='--', marker='o', markersize=4, alpha=0.7)
    if num_poisoned > 0:
        plt.plot(x_poisoned, poisoned_conds, label=f'Poisoned Models ({num_poisoned})', 
                 color='red', linestyle='-', marker='x', markersize=4, alpha=0.7)

    plt.xlabel("Model Index (Sequentially Processed)")
    plt.ylabel("Condition Number")
    plt.title("Global FC Layer Condition Number: Clean vs Poisoned Models")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    filename = "GLOBAL_condition_number_comparison.png"
    save_path = save_dir / filename
    plt.savefig(save_path)
    plt.close()
    
    print(f"\n📈 Global plot saved to:\n{save_path.resolve()}")

# --- Main Processing Function ---
def calculate_and_plot_global_condition_number():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    main_path = Path('models_all')
    plot_dir = Path('condition_number_plots')
    plot_dir.mkdir(exist_ok=True)

    if not main_path.exists():
        print(f"Error: '{main_path.resolve()}' not found.")
        return

    config_files = list(main_path.glob('*/*/config.json'))
    clean_conds = []
    poisoned_conds = []

    for config_path in tqdm.tqdm(config_files, desc="Processing all models"):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            state = config_data.get('py/state', {})
            is_poisoned = state.get('poisoned', False)

            model_path = config_path.parent / 'model.pt'
            if model_path.exists():
                fc_weights = load_fc_weights(model_path, device)
                cond_num = calculate_condition_number(fc_weights)

                if is_poisoned:
                    poisoned_conds.append(cond_num)
                else:
                    clean_conds.append(cond_num)
        except Exception as e:
            print(f" Error reading {config_path}: {e}")
            continue

    print("\n" + "="*70)
    print("📊 Condition number calculation finished.")
    print(f"  {len(clean_conds)} Clean models processed.")
    print(f"  {len(poisoned_conds)} Poisoned models processed.")
    print("="*70)

    plot_global_condition_number(clean_conds, poisoned_conds, plot_dir)
    print("\n✅ Processing complete.")

if __name__ == "__main__":
    calculate_and_plot_global_condition_number()
