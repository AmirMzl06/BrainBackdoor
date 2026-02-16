import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from cebra import CEBRA
from sklearn.metrics import r2_score

file_path = "hip/hippocampus_single_achilles.h5"
save_path = "./models"
os.makedirs(save_path, exist_ok=True)
window_size = 10 

with h5py.File(file_path, 'r') as f:
    cursor_vel = f['cursor/vel'][:]
    cursor_times = f['cursor/timestamp_indices_1s'][:]
    spike_times = f['spikes/timestamp_indices_1s'][:]
    spike_units = f['spikes/unit_index'][:]
    cursor_train_mask = f['cursor/train_mask'][:]
    cursor_test_mask = f['cursor/test_mask'][:]
    n_neurons = f['units/brain_area'].shape[0]

n_samples = len(cursor_vel)
neural = np.zeros((n_samples, n_neurons), dtype=np.float32)
indices = np.searchsorted(cursor_times, spike_times, side='right') - 1
valid = (indices >= 0) & (indices < n_samples)
for idx, unit in zip(indices[valid], spike_units[valid]):
    neural[idx, unit] += 1

neural_train, label_train = neural[cursor_train_mask], cursor_vel[cursor_train_mask]
neural_test, label_test = neural[cursor_test_mask], cursor_vel[cursor_test_mask]

cebra_model = CEBRA(model_architecture='offset10-model',
                    batch_size=512,
                    learning_rate=3e-4,
                    output_dimension=32,
                    max_iterations=5000, 
                    distance='cosine',
                    conditional='time_delta',
                    device='cuda_if_available',
                    verbose=True,
                    time_offsets=10)

cebra_model.fit(neural_train, label_train)
emb_train = cebra_model.transform(neural_train)
emb_test = cebra_model.transform(neural_test)

def create_sliding_windows(data, labels, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)

X_train_w, y_train_w = create_sliding_windows(emb_train, label_train, window_size)
X_test_w, y_test_w = create_sliding_windows(emb_test, label_test, window_size)

class Conv1dDecoder(nn.Module):
    def __init__(self, input_dim, window_size, output_dim):
        super(Conv1dDecoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * window_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        return self.fc_layers(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

y_min, y_max = y_train_w.min(axis=0), y_train_w.max(axis=0)
y_train_norm = (y_train_w - y_min) / (y_max - y_min)

model = Conv1dDecoder(input_dim=32, window_size=window_size, output_dim=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()

X_train_t = torch.FloatTensor(X_train_w).to(device)
y_train_t = torch.FloatTensor(y_train_norm).to(device)
X_test_t = torch.FloatTensor(X_test_w).to(device)

print("\n--- Training Conv1d Decoder ---")
for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_t)
    loss = criterion(preds, y_train_t)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t).cpu().numpy()
            test_preds_real = test_preds * (y_max - y_min) + y_min
            r2 = r2_score(y_test_w, test_preds_real, multioutput='uniform_average')
            print(f"Epoch {epoch+1} | Loss: {loss.item():.6f} | Test R2: {r2:.4f}")

print("\n--- Process Completed ---")


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

# # with h5py.File(file_path, 'r') as f:
# #     cursor_vel = f['cursor/vel'][:]
# #     cursor_times = f['cursor/timestamp_indices_1s'][:]
# #     spike_times = f['spikes/timestamp_indices_1s'][:]
# #     spike_units = f['spikes/unit_index'][:]
# #     cursor_train_mask = f['cursor/train_mask'][:]
# #     cursor_test_mask = f['cursor/test_mask'][:]
# #     n_neurons = f['units/brain_area'].shape[0]

# # n_samples = len(cursor_vel)
# # min_spike_len = min(len(spike_times), len(spike_units))
# # spike_times = spike_times[:min_spike_len]
# # spike_units = spike_units[:min_spike_len]

# # neural = np.zeros((n_samples, n_neurons), dtype=np.float32)
# # indices = np.searchsorted(cursor_times, spike_times, side='right') - 1
# # valid = (indices >= 0) & (indices < n_samples)
# # indices = indices[valid]
# # filtered_spike_units = spike_units[valid]

# # for idx, unit in zip(indices, filtered_spike_units):
# #     neural[idx, unit] += 1

# # print("Shape of neural matrix:", neural.shape)
# # print("Shape of cursor velocity:", cursor_vel.shape)

# # if cursor_train_mask.sum() > 0 and cursor_test_mask.sum() > 0:
# #     neural_train = neural[cursor_train_mask]
# #     neural_test = neural[cursor_test_mask]
# #     label_train = cursor_vel[cursor_train_mask]
# #     label_test = cursor_vel[cursor_test_mask]
# #     print("Using provided train/test masks.")
# # else:
# #     split_idx = int(n_samples * 0.8)
# #     neural_train = neural[:split_idx]
# #     neural_test = neural[split_idx:]
# #     label_train = cursor_vel[:split_idx]
# #     label_test = cursor_vel[split_idx:]
# #     print("Using random 80/20 split.")

# # max_iterations = 10000
# # output_dimension = 32
# # save_path = "./models"
# # os.makedirs(save_path, exist_ok=True)

# # cebra_pos_model = CEBRA(model_architecture='offset10-model',
# #                         batch_size=512,
# #                         learning_rate=3e-4,
# #                         temperature=1,
# #                         output_dimension=output_dimension,
# #                         max_iterations=max_iterations,
# #                         distance='cosine',
# #                         conditional='time_delta',
# #                         device='cuda_if_available',
# #                         verbose=True,
# #                         time_offsets=10)

# # cebra_pos_model.fit(neural_train, label_train)
# # cebra_pos_model.save(os.path.join(save_path, "cebra_pos_model.pt"))

# # cebra_pos_train = cebra_pos_model.transform(neural_train)
# # cebra_pos_test = cebra_pos_model.transform(neural_test)

# # class RobustDecoder(nn.Module):
# #     def __init__(self, input_dim):
# #         super(RobustDecoder, self).__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(input_dim, 128),
# #             nn.BatchNorm1d(128),
# #             nn.ReLU(),
# #             nn.Dropout(0.3),
# #             nn.Linear(128, 64),
# #             nn.ReLU(),
# #             nn.Linear(64, 2)
# #         )
# #     def forward(self, x):
# #         return self.net(x)

# # def train_decoder_optimized(emb_train, emb_test, label_train, label_test, epochs=500000, lr=0.001):
# #     y_train, y_test = label_train, label_test
# #     y_min, y_max = y_train.min(axis=0), y_train.max(axis=0)
# #     y_train_norm = (y_train - y_min) / (y_max - y_min)
    
# #     X_train = torch.FloatTensor(emb_train)
# #     y_train_target = torch.FloatTensor(y_train_norm)
# #     X_test = torch.FloatTensor(emb_test)
    
# #     model = RobustDecoder(input_dim=emb_train.shape[1])
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     model.to(device)
# #     X_train, y_train_target, X_test = X_train.to(device), y_train_target.to(device), X_test.to(device)
    
# #     criterion = nn.MSELoss()
# #     optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)
    
# #     for epoch in range(epochs):
# #         model.train()
# #         optimizer.zero_grad()
# #         loss = criterion(model(X_train), y_train_target)
# #         loss.backward()
# #         optimizer.step()
        
# #         if (epoch + 1) % 500 == 0:
# #             model.eval()
# #             with torch.no_grad():
# #                 pred_norm = model(X_test).cpu().numpy()
# #                 pred_real = pred_norm * (y_max - y_min) + y_min
# #                 r2 = r2_score(y_test, pred_real, multioutput='uniform_average')
# #             print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | R2: {r2:.4f}")
            
# #     return model, y_min, y_max

# # decoder_model, y_min, y_max = train_decoder_optimized(cebra_pos_train, cebra_pos_test, label_train, label_test)
# # print("--- Process Completed ---")
