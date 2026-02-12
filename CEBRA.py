import os
import sys
import numpy as np
import h5py
from types import SimpleNamespace
import matplotlib.pyplot as plt
import cebra
import cebra.datasets
from cebra import CEBRA
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

save_path = "/mnt/data/hossein/Hossein_workspace/nips_cetra/BrainBackdoor/code/BrainBackdoor/hip"
os.makedirs(save_path, exist_ok=True)
print(f"--- Directory '{save_path}' created (or already exists) ---")

file_path = os.path.join(save_path, "hippocampus_single_achilles.h5")
print(f"--- Loading dataset from {file_path} ---")

with h5py.File(file_path, 'r') as f:
    spikes = np.array(f['spikes']) 
    cursor = np.array(f['cursor'])     
    if cursor.ndim == 2 and cursor.shape[1] >= 1:
        position = cursor[:, 0]
    else:
        position = cursor.flatten()

print(f"Spikes shape: {spikes.shape}, Position shape: {position.shape}")

hippocampus_pos = SimpleNamespace()
hippocampus_pos.neural = spikes
hippocampus_pos.continuous_index = position.reshape(-1, 1)
hippocampus_pos.discrete_index = None

print("--- Dataset loaded manually successfully ---")

max_iterations = 10000
output_dimension = 32

def split_data(data, test_ratio=0.2):
    split_idx = int(len(data.neural) * (1 - test_ratio))
    neural_train = data.neural[:split_idx]
    neural_test = data.neural[split_idx:]

    if data.continuous_index is not None:
        label_train = data.continuous_index[:split_idx]
        label_test = data.continuous_index[split_idx:]
    elif data.discrete_index is not None:
        label_train = data.discrete_index[:split_idx]
        label_test = data.discrete_index[split_idx:]
    else:
        label_train = np.arange(split_idx).reshape(-1, 1)
        label_test = np.arange(split_idx, len(data.neural)).reshape(-1, 1)

    return (neural_train, neural_test, label_train, label_test)

neural_train, neural_test, label_train, label_test = split_data(hippocampus_pos, 0.2)
print(f"--- Data Split Done: Train {neural_train.shape}, Test {neural_test.shape} ---")

print("--- Training CEBRA Model (Position) ---")
cebra_pos_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='behavior',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

cebra_pos_model.fit(neural_train, label_train)
cebra_pos_model.save(os.path.join(save_path, "cebra_pos_model.pt"))
print("--- CEBRA Model Saved ---")

print("--- Training Shuffled CEBRA Model ---")
cebra_pos_shuffled_model = CEBRA(model_architecture='offset10-model',
                                 batch_size=512,
                                 learning_rate=3e-4,
                                 temperature=1,
                                 output_dimension=output_dimension,
                                 max_iterations=max_iterations,
                                 distance='cosine',
                                 conditional='behavior',
                                 device='cuda_if_available',
                                 verbose=True,
                                 time_offsets=10)

shuffled_pos = np.random.permutation(label_train)
cebra_pos_shuffled_model.fit(neural_train, shuffled_pos)
cebra_pos_shuffled_model.save(os.path.join(save_path, "cebra_pos_shuffled_model.pt"))
print("--- Shuffled Model Saved ---")

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
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_decoder_optimized(emb_train, emb_test, label_train, label_test, epochs=5000, lr=0.01):
    y_train_raw = label_train[:, 0] if len(label_train.shape) > 1 else label_train
    y_test_raw = label_test[:, 0] if len(label_test.shape) > 1 else label_test

    y_min, y_max = y_train_raw.min(), y_train_raw.max()
    y_train_norm = (y_train_raw - y_min) / (y_max - y_min)

    X_train = torch.FloatTensor(emb_train)
    y_train_target = torch.FloatTensor(y_train_norm).view(-1, 1)
    X_test = torch.FloatTensor(emb_test)

    model = RobustDecoder(input_dim=emb_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_train, y_train_target, X_test = X_train.to(device), y_train_target.to(device), X_test.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

    print(f"\n{'Epoch':<8} | {'Loss':<12} | {'R2 Score':<15}")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs_norm = model(X_train)
        loss = criterion(outputs_norm, y_train_target)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if (epoch + 1) % 500 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                pred_norm = model(X_test).cpu().numpy()
                pred_real = pred_norm * (y_max - y_min) + y_min
                current_r2 = r2_score(y_test_raw, pred_real)
            print(f"{epoch+1:<8} | {loss.item():<12.4f} | {current_r2:<15.4f}")

    return model, y_min, y_max

print("--- Training MLP Decoder ---")
decoder_model, y_min, y_max = train_decoder_optimized(cebra_pos_train, cebra_pos_test, label_train, label_test)

print("--- Process Completed ---")

#################################################
#################################################
#################################################

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

# save_path = "/mnt/data/hossein/Hossein_workspace/nips_cetra/BrainBackdoor/code/BrainBackdoor/hip"

# print(f"--- Directory '{save_path}' created ---")

# # print("--- Downloading and Loading Dataset ---")
# hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles', data_root=save_path)
# # print("--- Dataset Ready ---")

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
