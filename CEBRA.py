import os
import numpy as np
import h5py
import cebra
from cebra import CEBRA
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

file_path = "hip/hippocampus_single_achilles.h5"

with h5py.File(file_path, 'r') as f:
    cursor_vel = f['cursor/vel'][:]
    cursor_times = f['cursor/timestamp_indices_1s'][:]
    spike_times = f['spikes/timestamp_indices_1s'][:]
    spike_units = f['spikes/unit_index'][:]
    cursor_train_mask = f['cursor/train_mask'][:]
    cursor_test_mask = f['cursor/test_mask'][:]
    n_neurons = f['units/brain_area'].shape[0]

n_samples = len(cursor_vel)
min_spike_len = min(len(spike_times), len(spike_units))
spike_times = spike_times[:min_spike_len]
spike_units = spike_units[:min_spike_len]

neural = np.zeros((n_samples, n_neurons), dtype=np.float32)
indices = np.searchsorted(cursor_times, spike_times, side='right') - 1
valid = (indices >= 0) & (indices < n_samples)
indices = indices[valid]
filtered_spike_units = spike_units[valid]

for idx, unit in zip(indices, filtered_spike_units):
    neural[idx, unit] += 1

print("Shape of neural matrix:", neural.shape)
print("Shape of cursor velocity:", cursor_vel.shape)

if cursor_train_mask.sum() > 0 and cursor_test_mask.sum() > 0:
    neural_train = neural[cursor_train_mask]
    neural_test = neural[cursor_test_mask]
    label_train = cursor_vel[cursor_train_mask]
    label_test = cursor_vel[cursor_test_mask]
    print("Using provided train/test masks.")
else:
    split_idx = int(n_samples * 0.8)
    neural_train = neural[:split_idx]
    neural_test = neural[split_idx:]
    label_train = cursor_vel[:split_idx]
    label_test = cursor_vel[split_idx:]
    print("Using random 80/20 split.")

max_iterations = 10000
output_dimension = 32
save_path = "./models"
os.makedirs(save_path, exist_ok=True)

cebra_pos_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

cebra_pos_model.fit(neural_train, label_train)
cebra_pos_model.save(os.path.join(save_path, "cebra_pos_model.pt"))

cebra_pos_train = cebra_pos_model.transform(neural_train)
cebra_pos_test = cebra_pos_model.transform(neural_test)

class RobustDecoder(nn.Module):
    def __init__(self, input_dim):
        super(RobustDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_decoder_optimized(emb_train, emb_test, label_train, label_test, epochs=5000, lr=0.01):
    y_train, y_test = label_train, label_test
    y_min, y_max = y_train.min(axis=0), y_train.max(axis=0)
    y_train_norm = (y_train - y_min) / (y_max - y_min)
    
    X_train = torch.FloatTensor(emb_train)
    y_train_target = torch.FloatTensor(y_train_norm)
    X_test = torch.FloatTensor(emb_test)
    
    model = RobustDecoder(input_dim=emb_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_train, y_train_target, X_test = X_train.to(device), y_train_target.to(device), X_test.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train_target)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                pred_norm = model(X_test).cpu().numpy()
                pred_real = pred_norm * (y_max - y_min) + y_min
                r2 = r2_score(y_test, pred_real, multioutput='uniform_average')
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | R2: {r2:.4f}")
            
    return model, y_min, y_max

decoder_model, y_min, y_max = train_decoder_optimized(cebra_pos_train, cebra_pos_test, label_train, label_test)
print("--- Process Completed ---")

# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import cebra
# import cebra.datasets
# from cebra import CEBRA
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import r2_score

# hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles')

# max_iterations = 10000 
# output_dimension = 32 

# def split_data(data, test_ratio=0.2):
#     split_idx = int(len(data.neural) * (1 - test_ratio))
#     neural_train = data.neural[:split_idx]
#     neural_test = data.neural[split_idx:]

#     if data.continuous_index is not None:
#         label_train = data.continuous_index[:split_idx]
#         label_test = data.continuous_index[split_idx:]
#     elif data.discrete_index is not None:
#         label_train = data.discrete_index[:split_idx]
#         label_test = data.discrete_index[split_idx:]
#     else:
#         label_train = np.arange(split_idx).reshape(-1, 1)
#         label_test = np.arange(split_idx, len(data.neural)).reshape(-1, 1)
#         return (neural_train.numpy(), neural_test.numpy(), label_train, label_test)

#     return (neural_train.numpy(), neural_test.numpy(), label_train.numpy(), label_test.numpy())

# neural_train, neural_test, label_train, label_test = split_data(hippocampus_pos, 0.2)
# print(f"--- Data Split Done: Train {neural_train.shape}, Test {neural_test.shape} ---")

# print("--- Training CEBRA Model (Position) ---")
# cebra_pos_model = CEBRA(model_architecture='offset10-model',
#                         batch_size=512,
#                         learning_rate=3e-4,
#                         temperature=1,
#                         output_dimension=output_dimension,
#                         max_iterations=max_iterations,
#                         distance='cosine',
#                         conditional='behavior',
#                         device='cuda_if_available',
#                         verbose=True,
#                         time_offsets=10)

# cebra_pos_model.fit(neural_train, label_train)
# cebra_pos_model.save(os.path.join(save_path, "cebra_pos_model.pt"))
# print("--- CEBRA Model Saved ---")

# print("--- Training Shuffled CEBRA Model ---")
# cebra_pos_shuffled_model = CEBRA(model_architecture='offset10-model',
#                                  batch_size=512,
#                                  learning_rate=3e-4,
#                                  temperature=1,
#                                  output_dimension=output_dimension,
#                                  max_iterations=max_iterations,
#                                  distance='cosine',
#                                  conditional='behavior',
#                                  device='cuda_if_available',
#                                  verbose=True,
#                                  time_offsets=10)

# shuffled_pos = np.random.permutation(label_train)
# cebra_pos_shuffled_model.fit(neural_train, shuffled_pos)
# cebra_pos_shuffled_model.save(os.path.join(save_path, "cebra_pos_shuffled_model.pt"))
# print("--- Shuffled Model Saved ---")

# cebra_pos_train = cebra_pos_model.transform(neural_train)
# cebra_pos_test = cebra_pos_model.transform(neural_test)

# class RobustDecoder(nn.Module):
#     def __init__(self, input_dim):
#         super(RobustDecoder, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         return self.net(x)

# def train_decoder_optimized(emb_train, emb_test, label_train, label_test, epochs=5000, lr=0.01):
#     y_train_raw = label_train[:, 0] if len(label_train.shape) > 1 else label_train
#     y_test_raw = label_test[:, 0] if len(label_test.shape) > 1 else label_test

#     y_min, y_max = y_train_raw.min(), y_train_raw.max()
#     y_train_norm = (y_train_raw - y_min) / (y_max - y_min)

#     X_train = torch.FloatTensor(emb_train)
#     y_train_target = torch.FloatTensor(y_train_norm).view(-1, 1)
#     X_test = torch.FloatTensor(emb_test)

#     model = RobustDecoder(input_dim=emb_train.shape[1])
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     X_train, y_train_target, X_test = X_train.to(device), y_train_target.to(device), X_test.to(device)

#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

#     print(f"\n{'Epoch':<8} | {'Loss':<12} | {'R2 Score':<15}")
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         outputs_norm = model(X_train)
#         loss = criterion(outputs_norm, y_train_target)
#         loss.backward()
#         optimizer.step()
#         scheduler.step(loss)

#         if (epoch + 1) % 500 == 0 or epoch == epochs - 1:
#             model.eval()
#             with torch.no_grad():
#                 pred_norm = model(X_test).cpu().numpy()
#                 pred_real = pred_norm * (y_max - y_min) + y_min
#                 current_r2 = r2_score(y_test_raw, pred_real)
#             print(f"{epoch+1:<8} | {loss.item():<12.4f} | {current_r2:<15.4f}")

#     return model, y_min, y_max

# print("--- Training MLP Decoder ---")
# decoder_model, y_min, y_max = train_decoder_optimized(cebra_pos_train, cebra_pos_test, label_train, label_test)

# print("--- Process Completed ---")
