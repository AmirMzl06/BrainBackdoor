# import os
# import numpy as np
# import h5py
# import cebra
# from cebra import CEBRA
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import r2_score

# file_path = "hip/hippocampus_single_achilles.h5"

# with h5py.File(file_path, 'r') as f:
#     cursor_vel = f['cursor/vel'][:]
#     cursor_times = f['cursor/timestamp_indices_1s'][:]
#     spike_times = f['spikes/timestamp_indices_1s'][:]
#     spike_units = f['spikes/unit_index'][:]
#     cursor_train_mask = f['cursor/train_mask'][:]
#     cursor_test_mask = f['cursor/test_mask'][:]
#     n_neurons = f['units/brain_area'].shape[0]

# n_samples = len(cursor_vel)
# min_spike_len = min(len(spike_times), len(spike_units))
# spike_times = spike_times[:min_spike_len]
# spike_units = spike_units[:min_spike_len]

# neural = np.zeros((n_samples, n_neurons), dtype=np.float32)
# indices = np.searchsorted(cursor_times, spike_times, side='right') - 1
# valid = (indices >= 0) & (indices < n_samples)
# indices = indices[valid]
# filtered_spike_units = spike_units[valid]

# for idx, unit in zip(indices, filtered_spike_units):
#     neural[idx, unit] += 1

# print("Shape of neural matrix:", neural.shape)
# print("Shape of cursor velocity:", cursor_vel.shape)

# if cursor_train_mask.sum() > 0 and cursor_test_mask.sum() > 0:
#     neural_train = neural[cursor_train_mask]
#     neural_test = neural[cursor_test_mask]
#     label_train = cursor_vel[cursor_train_mask]
#     label_test = cursor_vel[cursor_test_mask]
#     print("Using provided train/test masks.")
# else:
#     split_idx = int(n_samples * 0.8)
#     neural_train = neural[:split_idx]
#     neural_test = neural[split_idx:]
#     label_train = cursor_vel[:split_idx]
#     label_test = cursor_vel[split_idx:]
#     print("Using random 80/20 split.")

# max_iterations = 10000
# output_dimension = 32
# save_path = "./models"
# os.makedirs(save_path, exist_ok=True)

# cebra_pos_model = CEBRA(model_architecture='offset10-model',
#                         batch_size=512,
#                         learning_rate=3e-3,
#                         temperature=2,
#                         output_dimension=output_dimension,
#                         max_iterations=max_iterations,
#                         distance='cosine',
#                         conditional='time_delta',
#                         device='cuda_if_available',
#                         verbose=True,
#                         time_offsets=10)

# cebra_pos_model.fit(neural_train, label_train)
# cebra_pos_model.save(os.path.join(save_path, "cebra_pos_model.pt"))

# cebra_pos_train = cebra_pos_model.transform(neural_train)
# cebra_pos_test = cebra_pos_model.transform(neural_test)

# class RobustDecoder(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.LayerNorm(64),
#             nn.ReLU(),
#             nn.Linear(64, 2)
#         )
#     def forward(self, x):
#         return self.net(x)


# def train_decoder_optimized(emb_train, emb_test, label_train, label_test, epochs=50000, lr=0.001):
#     y_train, y_test = label_train, label_test
#     y_min, y_max = y_train.min(axis=0), y_train.max(axis=0)
#     y_train_norm = (y_train - y_min) / (y_max - y_min)
    
#     X_train = torch.FloatTensor(emb_train)
#     y_train_target = torch.FloatTensor(y_train_norm)
#     X_test = torch.FloatTensor(emb_test)
    
#     model = RobustDecoder(input_dim=emb_train.shape[1])
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     X_train, y_train_target, X_test = X_train.to(device), y_train_target.to(device), X_test.to(device)
    
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)
    
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         loss = criterion(model(X_train), y_train_target)
#         loss.backward()
#         optimizer.step()
        
#         if (epoch + 1) % 500 == 0:
#             model.eval()
#             with torch.no_grad():
#                 pred_norm = model(X_test).cpu().numpy()
#                 pred_real = pred_norm * (y_max - y_min) + y_min
#                 r2 = r2_score(y_test, pred_real, multioutput='uniform_average')
#             print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | R2: {r2:.4f}")
            
#     return model, y_min, y_max

# decoder_model, y_min, y_max = train_decoder_optimized(cebra_pos_train, cebra_pos_test, label_train, label_test)
# print("--- Process Completed ---")





import os
import numpy as np
import h5py
import cebra
from cebra import CEBRA
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# --- 1. Load Data ---
file_path = "hip/hippocampus_single_achilles.h5"

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

print("Loading data...")
with h5py.File(file_path, 'r') as f:
    cursor_vel = f['cursor/vel'][:]
    cursor_times = f['cursor/timestamp_indices_1s'][:]
    spike_times = f['spikes/timestamp_indices_1s'][:]
    spike_units = f['spikes/unit_index'][:]
    cursor_train_mask = f['cursor/train_mask'][:]
    cursor_test_mask = f['cursor/test_mask'][:]
    if 'units/brain_area' in f:
        n_neurons = f['units/brain_area'].shape[0]
    else:
        n_neurons = int(spike_units.max()) + 1

n_samples = len(cursor_vel)
min_spike_len = min(len(spike_times), len(spike_units))
spike_times = spike_times[:min_spike_len]
spike_units = spike_units[:min_spike_len]

print("Binning spikes...")
neural = np.zeros((n_samples, n_neurons), dtype=np.float32)

indices = np.searchsorted(cursor_times, spike_times, side='right') - 1

valid_mask = (indices >= 0) & (indices < n_samples)
valid_indices = indices[valid_mask]
valid_units = spike_units[valid_mask]

np.add.at(neural, (valid_indices, valid_units), 1)

print(f"Neural matrix shape (Raw): {neural.shape}")

print("Smoothing neural data...")
neural_smooth = gaussian_filter1d(neural, sigma=2.0, axis=0)

if cursor_train_mask.sum() > 0 and cursor_test_mask.sum() > 0:
    print("Using provided train/test masks.")
    train_mask = cursor_train_mask.astype(bool)
    test_mask = cursor_test_mask.astype(bool)
    
    neural_train = neural_smooth[train_mask]
    neural_test = neural_smooth[test_mask]
    label_train = cursor_vel[train_mask]
    label_test = cursor_vel[test_mask]
else:
    print("Using random 80/20 split.")
    split_idx = int(n_samples * 0.8)
    neural_train = neural_smooth[:split_idx]
    neural_test = neural_smooth[split_idx:]
    label_train = cursor_vel[:split_idx]
    label_test = cursor_vel[split_idx:]

max_iterations = 15000 
output_dimension = 16  
save_path = "./models"
os.makedirs(save_path, exist_ok=True)

print("Initializing CEBRA...")
cebra_pos_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-3,
                        temperature=1,   
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

print("Fitting CEBRA model...")
cebra_pos_model.fit(neural_train, label_train)
cebra_pos_model.save(os.path.join(save_path, "cebra_pos_model.pt"))

print("Transforming data...")
embedding_train = cebra_pos_model.transform(neural_train)
embedding_test = cebra_pos_model.transform(neural_test)

print("\n--- Decoding Results ---")

knn = KNeighborsRegressor(n_neighbors=25, metric='cosine', weights='distance')
knn.fit(embedding_train, label_train)
knn_pred = knn.predict(embedding_test)
r2_knn = r2_score(label_test, knn_pred)

print(f"k-NN Regressor (k=25) R2 Score: {r2_knn:.4f}")

ridge = Ridge(alpha=1.0)
ridge.fit(embedding_train, label_train)
ridge_pred = ridge.predict(embedding_test)
r2_ridge = r2_score(label_test, ridge_pred)

print(f"Ridge Regression R2 Score:      {r2_ridge:.4f}")

# Optional: Save results
print("Process Completed.")
