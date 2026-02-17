import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
import cebra
from cebra import CEBRA

# --- تنظیمات اولیه ---
file_path = "hip/hippocampus_single_achilles.h5"
bin_size = 0.025 # 25ms bins
save_path = "./models"
os.makedirs(save_path, exist_ok=True)

# --- 1. لود کردن داده‌ها به سبک کد شما اما با اصلاح Binning ---
with h5py.File(file_path, 'r') as f:
    # لود داده‌های خام
    spike_times = f['spikes/timestamp_indices_1s'][:]
    spike_units = f['spikes/unit_index'][:]
    cursor_times = f['cursor/timestamp_indices_1s'][:]
    
    # پیدا کردن پوزیشن (به جای فقط سرعت)
    if 'cursor/pos' in f:
        labels = f['cursor/pos'][:]
    elif 'cursor/vel' in f:
        labels = f['cursor/vel'][:]
    
    n_neurons = int(spike_units.max()) + 1

# --- 2. تبدیل اسپایک‌ها به ماتریس عصبی (Binning صحیح) ---
# ایجاد بازه‌های زمانی ۲۵ میلی‌ثانیه‌ای
t_start = max(cursor_times[0], spike_times.min())
t_end = min(cursor_times[-1], spike_times.max())
bins = np.arange(t_start, t_end, bin_size)
n_bins = len(bins) - 1

neural = np.zeros((n_bins, n_neurons), dtype=np.float32)
for i in range(n_neurons):
    unit_spikes = spike_times[spike_units == i]
    counts, _ = np.histogram(unit_spikes, bins=bins)
    neural[:, i] = counts

# --- 3. تراز کردن لیبل‌ها با ماتریس عصبی ---
bin_centers = (bins[:-1] + bins[1:]) / 2
interpolator = interp1d(cursor_times, labels, axis=0, bounds_error=False, fill_value="extrapolate")
aligned_labels = interpolator(bin_centers).astype(np.float32)

print(f"Shape of neural matrix: {neural.shape}")
print(f"Shape of labels: {aligned_labels.shape}")

# --- 4. تقسیم‌بندی داده‌ها (80/20) ---
split_idx = int(len(neural) * 0.8)
neural_train, neural_test = neural[:split_idx], neural[split_idx:]
label_train, label_test = aligned_labels[:split_idx], aligned_labels[split_idx:]

# --- 5. آموزش مدل CEBRA ---
cebra_pos_model = CEBRA(
    model_architecture='offset10-model',
    batch_size=512,
    learning_rate=3e-4,
    temperature=1,
    output_dimension=32,
    max_iterations=5000,
    distance='cosine',
    conditional='time_delta',
    device='cuda_if_available',
    verbose=True,
    time_offsets=10,
    hybrid=True # بسیار مهم برای R2 مثبت
)

cebra_pos_model.fit(neural_train, label_train)
cebra_pos_model.save(os.path.join(save_path, "cebra_pos_model.pt"))

cebra_pos_train = cebra_pos_model.transform(neural_train)
cebra_pos_test = cebra_pos_model.transform(neural_test)

# --- 6. دکودر اختصاصی شما (RobustDecoder) ---
class RobustDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_decoder_optimized(emb_train, emb_test, label_train, label_test, epochs=5000, lr=0.001):
    y_train, y_test = label_train, label_test
    y_min, y_max = y_train.min(axis=0), y_train.max(axis=0)
    # جلوگیری از تقسیم بر صفر در صورت ثابت بودن یک ستون
    y_range = np.where((y_max - y_min) == 0, 1, y_max - y_min)
    y_train_norm = (y_train - y_min) / y_range
    
    X_train = torch.FloatTensor(emb_train)
    y_train_target = torch.FloatTensor(y_train_norm)
    X_test = torch.FloatTensor(emb_test)
    
    model = RobustDecoder(input_dim=emb_train.shape[1], output_dim=label_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_train, y_train_target, X_test = X_train.to(device), y_train_target.to(device), X_test.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
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
                pred_real = pred_norm * y_range + y_min
                r2 = r2_score(y_test, pred_real, multioutput='uniform_average')
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | R2: {r2:.4f}")
            
    return model, y_min, y_range

# اجرای نهایی
decoder_model, y_min, y_range = train_decoder_optimized(cebra_pos_train, cebra_pos_test, label_train, label_test)
print("--- Process Completed ---")



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
#                         learning_rate=3e-4,
#                         temperature=1,
#                         output_dimension=output_dimension,
#                         max_iterations=max_iterations,
#                         distance='cosine',
#                         conditional='time_delta',
#                         device='cuda_if_available',
#                         verbose=True,
#                         time_offsets=10,
#                         hybrid = True)

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
