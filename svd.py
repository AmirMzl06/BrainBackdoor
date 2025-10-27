import subprocess
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Install dependencies (optional, good practice) ---
try:
    print("Checking for 'timm' library...")
    import timm
    print("'timm' is already installed.")
except ImportError:
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
    Intelligently searches for 'classifier.1.weight' and 'fc.weight'.
    """
    try:
        model_state_dict = torch.load(model_path, map_location=device)
        
        # Handle cases where the full model is saved vs. just the state_dict
        if isinstance(model_state_dict, dict) and "state_dict" in model_state_dict:
            state_dict = model_state_dict["state_dict"]
        elif isinstance(model_state_dict, dict):
            state_dict = model_state_dict
        else:
            state_dict = model_state_dict.state_dict()

        fc_weights = None
        
        # Try to find the fully connected layer weights
        try:
            fc_weights = state_dict['classifier.1.weight']
        except KeyError:
            try:
                fc_weights = state_dict['fc.weight']
            except KeyError:
                print(f"⚠️ Error: Neither 'classifier.1.weight' nor 'fc.weight' found in model.")
                return None
        
        # The shape is typically [num_classes, num_features], which is what we need
        return fc_weights.detach().cpu().numpy()

    except Exception as e:
        print(f"❌ Error processing model {model_path}: {e}")
        return None

def calculate_svd_metrics(weights_np):
    """
    Calculates SVD-based metrics for the given weight matrix.
    Returns: A dictionary with calculated metrics.
    """
    if weights_np is None or weights_np.size == 0:
        return None

    # --- Metric 1: Spectral Concentration (based on Singular Values) ---
    # We only need the singular values, so we set compute_uv=False for efficiency
    singular_values = np.linalg.svd(weights_np, compute_uv=False)
    
    if np.sum(singular_values) > 1e-9: # Avoid division by zero
        spectral_concentration = singular_values[0] / np.sum(singular_values)
    else:
        spectral_concentration = 0.0

    # --- Metric 2: Outlier Score (based on L2 Norm Z-score) ---
    # In a typical FC layer of shape [num_classes, num_features], each row corresponds to a class.
    # We calculate the L2 norm for each class's weight vector (each row).
    class_norms = np.linalg.norm(weights_np, axis=1)
    
    mean_norm = np.mean(class_norms)
    std_norm = np.std(class_norms)
    
    if std_norm > 1e-9: # Avoid division by zero
        z_scores = (class_norms - mean_norm) / std_norm
        max_z_score = np.max(z_scores)
    else:
        max_z_score = 0.0
        
    return {
        "spectral_concentration": spectral_concentration,
        "max_z_score": max_z_score,
        "singular_values": singular_values
    }

def plot_singular_values(singular_values, save_path):
    """
    Plots the singular values spectrum (scree plot).
    """
    plt.figure(figsize=(12, 7))
    x_axis = np.arange(1, len(singular_values) + 1)
    
    plt.plot(x_axis, singular_values, marker='o', linestyle='-', color='b', label='Singular Values')
    plt.bar(x_axis, singular_values, color='skyblue', alpha=0.6, label='Magnitude')
    
    plt.xlabel("Singular Value Index (Ordered from largest to smallest)")
    plt.ylabel("Magnitude")
    plt.title("Singular Value Spectrum of the FC Layer Weights")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log') # Log scale is often better for visualizing the drop-off
    
    plt.savefig(save_path)
    plt.close()
    print(f"\n📈 Plot saved to file:\n{save_path.resolve()}")

# --- Main Execution ---

def analyze_single_model_with_svd():
    """
    Main function to analyze a single model file using SVD.
    """
    # --- Define the specific path to your model ---
    model_path = Path('models_all/01/id-00000001/model.pt')
    
    # --- Directory to save the output plot ---
    output_dir = Path('svd_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶️ Using device: '{device}' for processing.")
    
    # Check if the model file exists
    if not model_path.exists():
        print(f"❌ FATAL ERROR: Model file not found at '{model_path.resolve()}'")
        return

    print(f"\nLoading FC weights from: {model_path.name}...")
    fc_weights = load_fc_weights(model_path, device)
    
    if fc_weights is None:
        print("\nCould not proceed with analysis.")
        return
        
    print(f"✅ FC weights loaded successfully. Shape: {fc_weights.shape}")
    
    print("\nCalculating SVD-based metrics...")
    metrics = calculate_svd_metrics(fc_weights)
    
    if metrics:
        print("\n" + "="*50)
        print("📊 SVD Analysis Results 📊")
        print("="*50)
        
        # Higher values for these metrics can indicate a backdoor attack
        print(f"  - Spectral Concentration: {metrics['spectral_concentration']:.4f}")
        print(f"  - Max Z-Score (Outlier Score): {metrics['max_z_score']:.4f}")
        
        print("="*50)
        
        # Plot the results
        plot_save_path = output_dir / f"{model_path.parent.name}_svd_spectrum.png"
        plot_singular_values(metrics['singular_values'], plot_save_path)
    else:
        print("❌ Failed to calculate metrics.")

    print("\n✅ Processing complete.")


if __name__ == "__main__":
    analyze_single_model_with_svd()
