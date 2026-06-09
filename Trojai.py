# import subprocess
# import sys

# def upgrade_torch():
#     try:
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "torch"])
#         print("PyTorch has been upgraded successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"Error upgrading PyTorch: {e}")

# upgrade_torch()

# import json
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from scipy.stats import pearsonr
# from scipy.spatial.distance import pdist
# import tqdm
# print(torch.__version__)

# def _find_grid_dims(n):
#     if n <= 0:
#         raise ValueError("num of nourons must be positive")
#     for i in range(int(np.sqrt(n)), 0, -1):
#         if n % i == 0:
#             return i, n // i
#     return 1, n

# def compute_topolm_spatial_loss(hidden_states: np.ndarray):
#     if hidden_states.ndim != 2:
#         raise ValueError(f"Error: Input must be 2D, but got {hidden_states.ndim}D shape")

#     seq_len, hidden_dim = hidden_states.shape
    
#     if hidden_dim < 2:
#         return None 

#     grid_h, grid_w = _find_grid_dims(hidden_dim)
#     coords = np.array([(i, j) for i in range(grid_h) for j in range(grid_w)])

#     spatial_distances = pdist(coords, metric='chebyshev')
#     d_k = 1.0 / (1.0 + spatial_distances)

#     correlation_matrix = np.corrcoef(hidden_states, rowvar=False)
#     r_k = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]

#     valid_indices = ~np.isnan(r_k)
#     r_k = r_k[valid_indices]
#     d_k = d_k[valid_indices]

#     if len(r_k) < 2:
#         return None 

#     final_corr, _ = pearsonr(r_k, d_k)
#     if np.isnan(final_corr):
#         return None

#     spatial_loss = 0.5 * (1 - final_corr)
#     return spatial_loss

# def calculate_all_layers_loss(all_layer_tensors: list):
#     all_losses = []

#     for i, layer_tensor in enumerate(all_layer_tensors):
#         if not isinstance(layer_tensor, torch.Tensor):
#             raise TypeError("Error: Input list must contain torch.Tensors")

#         if layer_tensor.ndim < 2:
#             all_losses.append(None)
#             continue
        
#         if layer_tensor.ndim > 2:
#             try:
#                 hidden_states_np = layer_tensor.view(layer_tensor.shape[0], -1).cpu().numpy()
#             except Exception as e:
#                 all_losses.append(None)
#                 continue
#         else:
#             hidden_states_np = layer_tensor.cpu().numpy()

#         try:
#             loss = compute_topolm_spatial_loss(hidden_states_np)
#             # print(f"Layer {i} loss: {loss}")
#             all_losses.append(loss)
#         except Exception as e:
#             all_losses.append(None)
            
#     return all_losses

# def process_models_and_plot():
#     main_path = Path('models_all')
#     if not main_path.exists():
#         print(f"Error: Directory not found: {main_path.resolve()}")
#         print("Please run this script in the same directory as 'models_all'")
#         return

#     results = []
    
#     config_files = sorted(list(main_path.glob('*/*/config.json')))
#     print(f"Found {len(config_files)} models to process...")

#     for config_path in tqdm.tqdm(config_files, desc="Processing models"):
#         model_dir = config_path.parent
#         model_id = model_dir.name
#         model_path = model_dir / 'model.pt'

#         if not model_path.exists():
#             print(f"Warning: model.pt not found for {model_id}, skipping.")
#             continue

#         try:
#             with open(config_path, 'r') as f:
#                 config_data = json.load(f)
#             is_poisoned = config_data.get('py/state', {}).get('poisoned', False)
#         except Exception as e:
#             print(f"Error reading config {config_path}: {e}, skipping.")
#             continue

#         try:
#             model = torch.load(model_path, map_location="cpu", weights_only=False)
#             state_dict = model.state_dict()
#             weight_tensors = list(state_dict.values())
            
#         except Exception as e:
#             print(f"Error loading model {model_path}: {e}, skipping.")
#             continue
            
#         try:
#             layer_losses = calculate_all_layers_loss(weight_tensors)
#             valid_losses = [l for l in layer_losses if l is not None]
          
#             if valid_losses:
#                  results.append({
#                     'id': model_id,
#                     'poisoned': is_poisoned,
#                     'layer_losses': valid_losses  
#                 })
                
#         except Exception as e:
#             print(f"Error calculating loss for {model_id}: {e}, skipping.")
#             continue
            
#     print("Processing complete. Generating plot...")

    
#     clean_models_losses = [r['layer_losses'] for r in results if not r['poisoned'] and r['layer_losses']]
#     poisoned_models_losses = [r['layer_losses'] for r in results if r['poisoned'] and r['layer_losses']]

#     if not clean_models_losses and not poisoned_models_losses:
#         print("No valid data to plot.")
#         return

#     plt.figure(figsize=(12, 7))
    
  
#     for i, losses_list in enumerate(clean_models_losses):
#         label = f'Clean Models (n={len(clean_models_losses)})' if i == 0 else None
#         plt.plot(range(len(losses_list)), losses_list, color='blue', alpha=0.4, label=label)
    
#     for i, losses_list in enumerate(poisoned_models_losses):
#         label = f'Poisoned Models (n={len(poisoned_models_losses)})' if i == 0 else None
#         plt.plot(range(len(losses_list)), losses_list, color='red', alpha=0.4, label=label)
    
#     plt.title('Topological Loss per Layer (Weights)')
#     plt.xlabel('Layer Index')
#     plt.ylabel('Topological Loss')
#     plt.legend()
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
    
#     plot_filename = "model_loss_per_layer.png"
#     plt.savefig(plot_filename)
#     print(f"Plot saved as {plot_filename}")
    
#     plt.show()


# if __name__ == "__main__":
#     try:
#         import matplotlib
#         import scipy
#         import tqdm
#     except ImportError:
#         print("Please install required packages: pip install torch numpy matplotlib scipy tqdm")
#     else:
#         process_models_and_plot()



python TANR.py 
Running on: cuda
--- Training Baseline Model ---
Epoch 001/80 | Loss: 0.8226 | R2: 0.2905
Epoch 005/80 | Loss: 0.1287 | R2: 0.7045
Epoch 010/80 | Loss: 0.0498 | R2: 0.7151
Epoch 015/80 | Loss: 0.0263 | R2: 0.6966
Epoch 020/80 | Loss: 0.0173 | R2: 0.6643
Epoch 025/80 | Loss: 0.0129 | R2: 0.6494
Epoch 030/80 | Loss: 0.0108 | R2: 0.6319
Epoch 035/80 | Loss: 0.0081 | R2: 0.6352
Epoch 040/80 | Loss: 0.0062 | R2: 0.6176
Epoch 045/80 | Loss: 0.0053 | R2: 0.6172
Epoch 050/80 | Loss: 0.0088 | R2: 0.6079
Epoch 055/80 | Loss: 0.0047 | R2: 0.6184
Epoch 060/80 | Loss: 0.0053 | R2: 0.6139
Epoch 065/80 | Loss: 0.0055 | R2: 0.6126
Epoch 070/80 | Loss: 0.0045 | R2: 0.6024
Epoch 075/80 | Loss: 0.0045 | R2: 0.6083
Epoch 080/80 | Loss: 0.0033 | R2: 0.5934
Baseline Time: 6.38s | Final R2: 0.5934
--- Training tanr_l001 ---
Epoch 001/80 | Loss: 5.8490 | R2: 0.1846
Epoch 005/80 | Loss: 4.1791 | R2: 0.6456
Epoch 010/80 | Loss: 3.9980 | R2: 0.6903
Epoch 015/80 | Loss: 3.9222 | R2: 0.6775
Epoch 020/80 | Loss: 3.8823 | R2: 0.6718
Epoch 025/80 | Loss: 3.8592 | R2: 0.6681
Epoch 030/80 | Loss: 3.8453 | R2: 0.6464
Epoch 035/80 | Loss: 3.8344 | R2: 0.6629
Epoch 040/80 | Loss: 3.8264 | R2: 0.6536
Epoch 045/80 | Loss: 3.8209 | R2: 0.6404
Epoch 050/80 | Loss: 3.8164 | R2: 0.6615
Epoch 055/80 | Loss: 3.8113 | R2: 0.6531
Epoch 060/80 | Loss: 3.8090 | R2: 0.6525
Epoch 065/80 | Loss: 3.8049 | R2: 0.6559
Epoch 070/80 | Loss: 3.8025 | R2: 0.6447
Epoch 075/80 | Loss: 3.7992 | R2: 0.6481
Epoch 080/80 | Loss: 3.7992 | R2: 0.6522
tanr_l001 Time: 11.53s | Final R2: 0.6522
Baseline Latent Space PCA explained variance: 0.8051
tanr_l001 Latent Space PCA explained variance: 0.3714
--- Training tanr_l005 ---
Epoch 001/80 | Loss: 4.1115 | R2: 0.3261
Epoch 005/80 | Loss: 3.3225 | R2: 0.6866
Epoch 010/80 | Loss: 3.1602 | R2: 0.7123
Epoch 015/80 | Loss: 3.1039 | R2: 0.6855
Epoch 020/80 | Loss: 3.0775 | R2: 0.6794
Epoch 025/80 | Loss: 3.0599 | R2: 0.6702
Epoch 030/80 | Loss: 3.0501 | R2: 0.6744
Epoch 035/80 | Loss: 3.0418 | R2: 0.6722
Epoch 040/80 | Loss: 3.0344 | R2: 0.6583
Epoch 045/80 | Loss: 3.0318 | R2: 0.6417
Epoch 050/80 | Loss: 3.0274 | R2: 0.6393
Epoch 055/80 | Loss: 3.0228 | R2: 0.6318
Epoch 060/80 | Loss: 3.0199 | R2: 0.6438
Epoch 065/80 | Loss: 3.0168 | R2: 0.6245
Epoch 070/80 | Loss: 3.0145 | R2: 0.6110
Epoch 075/80 | Loss: 3.0122 | R2: 0.5903
Epoch 080/80 | Loss: 3.0120 | R2: 0.5933
tanr_l005 Time: 11.37s | Final R2: 0.5933
Baseline Latent Space PCA explained variance: 0.8051
tanr_l005 Latent Space PCA explained variance: 0.3866
--- Training tanr_l010 ---
Epoch 001/80 | Loss: 5.7171 | R2: 0.2733
Epoch 005/80 | Loss: 4.2213 | R2: 0.6516
Epoch 010/80 | Loss: 4.0087 | R2: 0.6645
Epoch 015/80 | Loss: 3.9285 | R2: 0.6364
Epoch 020/80 | Loss: 3.8917 | R2: 0.6331
Epoch 025/80 | Loss: 3.8716 | R2: 0.6381
Epoch 030/80 | Loss: 3.8605 | R2: 0.6168
Epoch 035/80 | Loss: 3.8495 | R2: 0.6159
Epoch 040/80 | Loss: 3.8428 | R2: 0.6384
Epoch 045/80 | Loss: 3.8383 | R2: 0.6329
Epoch 050/80 | Loss: 3.8334 | R2: 0.6323
Epoch 055/80 | Loss: 3.8285 | R2: 0.6168
Epoch 060/80 | Loss: 3.8258 | R2: 0.6290
Epoch 065/80 | Loss: 3.8228 | R2: 0.6183
Epoch 070/80 | Loss: 3.8208 | R2: 0.6284
Epoch 075/80 | Loss: 3.8182 | R2: 0.6222
Epoch 080/80 | Loss: 3.8187 | R2: 0.6256
tanr_l010 Time: 11.44s | Final R2: 0.6256
Baseline Latent Space PCA explained variance: 0.8051
tanr_l010 Latent Space PCA explained variance: 0.4274

Final Results:
baseline: 0.5934
tanr_l001: 0.6522
tanr_l005: 0.5933
tanr_l010: 0.6256
