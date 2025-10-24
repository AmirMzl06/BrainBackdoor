import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
from tqdm import tqdm

def _find_grid_dims(n):
    if n <= 0:
        raise ValueError("num of nourons must be positive")
    for i in range(int(np.sqrt(n)), 0, -1):
        if n % i == 0:
            return i, n // i
    return 1, n

def compute_topolm_spatial_loss(hidden_states: np.ndarray):
    if hidden_states.ndim != 2:
        raise ValueError(f"Error: Input must be 2D, but got {hidden_states.ndim}D shape")

    seq_len, hidden_dim = hidden_states.shape
    
    if hidden_dim < 2:
        return None 

    grid_h, grid_w = _find_grid_dims(hidden_dim)
    coords = np.array([(i, j) for i in range(grid_h) for j in range(grid_w)])

    spatial_distances = pdist(coords, metric='chebyshev')
    d_k = 1.0 / (1.0 + spatial_distances)

    correlation_matrix = np.corrcoef(hidden_states, rowvar=False)
    r_k = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]

    valid_indices = ~np.isnan(r_k)
    r_k = r_k[valid_indices]
    d_k = d_k[valid_indices]

    if len(r_k) < 2:
        return None 

    final_corr, _ = pearsonr(r_k, d_k)
    if np.isnan(final_corr):
        return None

    spatial_loss = 0.5 * (1 - final_corr)
    return spatial_loss

def calculate_all_layers_loss(all_layer_tensors: list):
    all_losses = []

    for i, layer_tensor in enumerate(all_layer_tensors):
        if not isinstance(layer_tensor, torch.Tensor):
            raise TypeError("Error: Input list must contain torch.Tensors")

        if layer_tensor.ndim < 2:
            all_losses.append(None)
            continue
        
        if layer_tensor.ndim > 2:
            try:
                hidden_states_np = layer_tensor.view(layer_tensor.shape[0], -1).cpu().numpy()
            except Exception as e:
                all_losses.append(None)
                continue
        else:
            hidden_states_np = layer_tensor.cpu().numpy()

        try:
            loss = compute_topolm_spatial_loss(hidden_states_np)
            all_losses.append(loss)
        except Exception as e:
            all_losses.append(None)
            
    return all_losses

def process_models_and_plot():
    main_path = Path('models_all')
    if not main_path.exists():
        print(f"Error: Directory not found: {main_path.resolve()}")
        print("Please run this script in the same directory as 'models_all'")
        return

    results = []
    
    config_files = sorted(list(main_path.glob('*/*/config.json')))
    print(f"Found {len(config_files)} models to process...")

    for config_path in tqdm(config_files, desc="Processing models"):
        model_dir = config_path.parent
        model_id = model_dir.name
        model_path = model_dir / 'model.pt'

        if not model_path.exists():
            print(f"Warning: model.pt not found for {model_id}, skipping.")
            continue

        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            is_poisoned = config_data.get('py/state', {}).get('poisoned', False)
        except Exception as e:
            print(f"Error reading config {config_path}: {e}, skipping.")
            continue

        try:
            model = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = model.state_dict()
            weight_tensors = list(state_dict.values())
            
        except Exception as e:
            print(f"Error loading model {model_path}: {e}, skipping.")
            continue
            
        try:
            layer_losses = calculate_all_layers_loss(weight_tensors)
            valid_losses = [l for l in layer_losses if l is not None]
            
            if valid_losses:
                avg_loss = np.mean(valid_losses)
            else:
                avg_loss = np.nan 
                
        except Exception as e:
            print(f"Error calculating loss for {model_id}: {e}, skipping.")
            continue
            
        results.append({
            'id': model_id,
            'poisoned': is_poisoned,
            'avg_loss': avg_loss
        })

    print("Processing complete. Generating plot...")

    clean_avg_losses = [r['avg_loss'] for r in results if not r['poisoned'] and not np.isnan(r['avg_loss'])]
    poisoned_avg_losses = [r['avg_loss'] for r in results if r['poisoned'] and not np.isnan(r['avg_loss'])]

    if not clean_avg_losses and not poisoned_avg_losses:
        print("No valid data to plot.")
        return

    plt.figure(figsize=(12, 7))
    
    all_losses = clean_avg_losses + poisoned_avg_losses
    bins = np.linspace(min(all_losses), max(all_losses), 40) 

    plt.hist(clean_avg_losses, bins=bins, color='red', alpha=0.7, label=f'Clean Models (n={len(clean_avg_losses)})', density=True)
    plt.hist(poisoned_avg_losses, bins=bins, color='blue', alpha=0.7, label=f'Poisoned Models (n={len(poisoned_avg_losses)})', density=True)
    
    plt.title('Distribution of Average Topological Loss (Weights)')
    plt.xlabel('Average Topological Loss')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_filename = "model_loss_distribution.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    
    plt.show()


if __name__ == "__main__":
    try:
        import matplotlib
        import scipy
        import tqdm
    except ImportError:
        print("Please install required packages: pip install torch numpy matplotlib scipy tqdm")
    else:
        process_models_and_plot()
