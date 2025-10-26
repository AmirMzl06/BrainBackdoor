import subprocess
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm
from collections import defaultdict

# --- Install dependencies ---
try:
    print("Installing/Updating 'timm'...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "timm"])
except Exception as e:
    print(f"Error installing 'timm': {e}")
    sys.exit(1)

# --- Constants ---
SPARSITY_THRESHOLD = 0.01

# --- Helper Functions ---

def load_fc_weights(model_path, device):
    """
    Loads the model from the given path and extracts the FC layer weights.
    Intelligently searches for both 'classifier.1.weight' and 'fc.weight'.
    """
    try:
        model_state_dict = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle cases where the full model is saved vs. just the state_dict
        if isinstance(model_state_dict, dict) and "state_dict" in model_state_dict:
            state_dict = model_state_dict["state_dict"]
        elif isinstance(model_state_dict, dict):
            state_dict = model_state_dict
        else:
            # Assuming a full model object was saved
            state_dict = model_state_dict.state_dict()

        fc_weights = None
        
        # Try-catch logic as requested
        try:
            # First, try the common layer name in torchvision models
            fc_weights = state_dict['classifier.1.weight']
        except KeyError:
            try:
                # If not found, try the common layer name in timm models
                fc_weights = state_dict['fc.weight']
            except KeyError:
                print(f"⚠️ Error: Neither 'classifier.1.weight' nor 'fc.weight' found in model {model_path.parent.name}.")
                return None
        
        return fc_weights.detach().cpu().numpy()

    except Exception as e:
        print(f" Error processing model {model_path.parent.name}: {e}")
        return None

def calculate_sparsity(weights_np, threshold):
    """
    Calculates the sparsity percentage for the given numpy array of weights.
    """
    if weights_np is None:
        return 0.0
        
    weights_flat = weights_np.flatten()
    total_params = weights_flat.size
    
    if total_params == 0:
        return 0.0

    # Calculate the number of weights whose absolute value is less than the threshold
    zero_params = np.sum(np.abs(weights_flat) < threshold)
    sparsity_percent = (zero_params / total_params) * 100
    return sparsity_percent

# -----------------------------------------------------------------
# --- New function for plotting the global graph ---
# -----------------------------------------------------------------
def plot_global_sparsity_graph(clean_sparsities, poisoned_sparsities, save_dir):
    """
    Plots a single line graph comparing the sparsity of *all* Clean vs. *all* Poisoned models.
    """
    
    num_clean = len(clean_sparsities)
    num_poisoned = len(poisoned_sparsities)

    if num_clean == 0 and num_poisoned == 0:
        print("No data found to plot the global graph.")
        return

    # X-axis for each plot, from 1 to N (number of models)
    x_clean = np.arange(1, num_clean + 1)
    x_poisoned = np.arange(1, num_poisoned + 1)

    plt.figure(figsize=(15, 8))
    
    # Use markers for clarity, and lighter lines
    if num_clean > 0:
        plt.plot(x_clean, clean_sparsities, 
                 label=f'Clean Models ({num_clean} models)', 
                 color='blue', 
                 linestyle='--', 
                 marker='o', 
                 markersize=4, 
                 alpha=0.7)
    
    if num_poisoned > 0:
        plt.plot(x_poisoned, poisoned_sparsities, 
                 label=f'Poisoned Models ({num_poisoned} models)', 
                 color='red', 
                 linestyle='-', 
                 marker='x', 
                 markersize=4, 
                 alpha=0.7)
    
    plt.xlabel("Model Index (Sequentially Processed)")
    plt.ylabel("Sparsity (%)")
    plt.title("Global FC Layer Sparsity Comparison: All Clean vs. All Poisoned Models")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    filename = "GLOBAL_sparsity_comparison_For_0.01.png"
    save_path = save_dir / filename
    
    plt.savefig(save_path)
    plt.close()
    
    print(f"\n📈 Global plot saved to file:\n{save_path.resolve()}")

# --- Main Processing Function ---

def calculate_and_plot_global_sparsity():
    """
    Scans all models, calculates their FC layer sparsity, and plots one global graph.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: '{device}' for processing.")

    main_path = Path('models_all')
    sparsity_plots_dir = Path('sparsity_plots')
    sparsity_plots_dir.mkdir(exist_ok=True)

    if not main_path.exists():
        print(f" Error: Directory '{main_path.resolve()}' not found.")
        return

    print("Scanning all models...")
    config_files = list(main_path.glob('*/*/config.json'))
    
    # --- Change: Instead of grouping, just create two global lists ---
    all_clean_sparsities = []
    all_poisoned_sparsities = []

    for config_path in tqdm.tqdm(config_files, desc="Processing all models"):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            state = config_data.get('py/state', {})
            is_poisoned = state.get('poisoned', False)
            
            model_path = config_path.parent / 'model.pt'
            
            if model_path.exists():
                # Load the FC weights
                fc_weights = load_fc_weights(model_path, device)
                
                if fc_weights is not None:
                    # Calculate sparsity
                    sparsity = calculate_sparsity(fc_weights, SPARSITY_THRESHOLD)
                    
                    # Add to the corresponding list
                    if is_poisoned:
                        all_poisoned_sparsities.append(sparsity)
                    else:
                        all_clean_sparsities.append(sparsity)
                else:
                    print(f"  Skipping model {config_path.parent.name} due to weight loading error.")

        except Exception as e:
            print(f" Error reading {config_path}: {e}")
            continue
    
    print("\n" + "="*70)
    print(" 📊 Sparsity calculation finished.")
    print(f"   {len(all_clean_sparsities)} Clean models processed.")
    print(f"   {len(all_poisoned_sparsities)} Poisoned models processed.")
    print("="*70)

    # --- Step 2: Plot the global graph ---
    plot_global_sparsity_graph(
        all_clean_sparsities,
        all_poisoned_sparsities,
        sparsity_plots_dir
    )

    print("\n✅ Processing complete.")


if __name__ == "__main__":
    calculate_and_plot_global_sparsity()
