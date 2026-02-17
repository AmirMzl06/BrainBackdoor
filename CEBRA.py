import os
import sys
import numpy as np
import h5py
import torch
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import cebra
from cebra import CEBRA

# ==========================================
# 1. Configuration & Path
# ==========================================
FILE_PATH = "hip/hippocampus_single_achilles.h5"  # مسیر فایل خود را چک کنید
BIN_SIZE = 0.025  # 25 ms bin size (Standard for rat hippocampus)
OUTPUT_DIM = 32   # Embedding dimension
MAX_ITER = 5000   # تعداد دورهای آموزش
BATCH_SIZE = 512

# ==========================================
# 2. Custom Data Loader (Correct Way)
# ==========================================
def load_and_process_data(file_path, bin_size):
    print(f"--- Loading file: {file_path} ---")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with h5py.File(file_path, 'r') as f:
        # --- A. Load Spikes ---
        # بررسی نام‌های مختلف برای اسپایک‌ها
        if 'spikes/timestamp_indices_1s' in f:
            spike_times = f['spikes/timestamp_indices_1s'][:]
            spike_units = f['spikes/unit_index'][:]
        else:
            raise KeyError("Could not find spike timestamps in the file.")

        # --- B. Load Continuous Behavior (Position/Direction) ---
        # اولویت با Position است چون نورون‌های مکان (Place Cells) به مکان حساس‌اند
        raw_pos = None
        
        # چک کردن کلیدهای مختلف که ممکن است در فایل باشند
        possible_pos_keys = ['cursor/pos', 'cursor/position', 'agent/pos']
        for key in possible_pos_keys:
            if key in f:
                raw_pos = f[key][:]
                print(f"-> Found position data in: {key}")
                break
        
        # اگر پوزیشن پیدا نشد، سراغ X و Y جداگانه می‌رویم
        if raw_pos is None and 'cursor/x' in f:
            px = f['cursor/x'][:]
            py = f['cursor/y'][:]
            raw_pos = np.stack([px, py], axis=1)
            print("-> Constructed position from cursor/x and cursor/y")

        # اگر باز هم پیدا نشد، سراغ سرعت می‌رویم (کمترین دقت)
        if raw_pos is None:
            print("!! WARNING: Position not found. Using Velocity. R2 might be lower.")
            raw_pos = f['cursor/vel'][:]

        # زمان‌های مربوط به حرکت (Cursor)
        if 'cursor/timestamp_indices_1s' in f:
            cursor_times = f['cursor/timestamp_indices_1s'][:]
        else:
            # اگر زمان نداشت، فرض می‌کنیم هم‌زمان با اسپایک‌هاست
            cursor_times = np.linspace(spike_times.min(), spike_times.max(), len(raw_pos))

    # --- C. Binning Spikes (The Logic of CEBRA) ---
    # ساختن پنجره‌های زمانی
    start_time = max(cursor_times.min(), spike_times.min())
    end_time = min(cursor_times.max(), spike_times.max())
    bins = np.arange(start_time, end_time, bin_size)
    n_bins = len(bins) - 1
    
    n_neurons = int(spike_units.max()) + 1
    neural_matrix = np.zeros((n_bins, n_neurons), dtype=np.float32)
    
    print(f"-> Binning spikes into {n_bins} time windows of {bin_size}s...")
    for unit_id in range(n_neurons):
        unit_spikes = spike_times[spike_units == unit_id]
        counts, _ = np.histogram(unit_spikes, bins=bins)
        neural_matrix[:, unit_id] = counts

    # --- D. Aligning Behavior to Bins ---
    # درونیابی (Interpolation) برای اینکه مکان دقیقاً وسط هر بازه زمانی محاسبه شود
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # ساخت تابع Interpolator
    interpolator = interp1d(cursor_times, raw_pos, axis=0, bounds_error=False, fill_value="extrapolate")
    continuous_index = interpolator(bin_centers)

    return neural_matrix, continuous_index

# ==========================================
# 3. Main Execution Pipeline
# ==========================================

# 1. Load Data
try:
    neural, behavior = load_and_process_data(FILE_PATH, BIN_SIZE)
    print(f"Neural Shape: {neural.shape} (Time x Neurons)")
    print(f"Behavior Shape: {behavior.shape} (Time x Features)")
except Exception as e:
    print(f"CRITICAL ERROR LOADING DATA: {e}")
    sys.exit(1)

# 2. Prepare Data for CEBRA
# Split train/test (80/20 standard)
split_idx = int(len(neural) * 0.8)
neural_train, neural_test = neural[:split_idx], neural[split_idx:]
label_train, label_test = behavior[:split_idx], behavior[split_idx:]

# 3. Define CEBRA Model
# نکته مهم: استفاده از hybrid=True و distance='cosine'
cebra_model = CEBRA(
    model_architecture='offset10-model',
    batch_size=BATCH_SIZE,
    learning_rate=3e-4,     # نرخ یادگیری استاندارد
    temperature=1,          # دمای پایین‌تر برای پایداری
    output_dimension=OUTPUT_DIM,
    max_iterations=MAX_ITER,
    distance='cosine',
    conditional='time_delta',
    device='cuda_if_available',
    verbose=True,
    time_offsets=10,
    hybrid=True             # <--- این پارامتر برای استفاده از لیبل‌ها حیاتی است
)

print("\n--- Starting CEBRA Training ---")
cebra_model.fit(neural_train, label_train)
print("--- Training Finished ---")

# 4. Transform (Get Embeddings)
embedding_train = cebra_model.transform(neural_train)
embedding_test = cebra_model.transform(neural_test)

# ==========================================
# 4. Decoding & Evaluation (KNN)
# ==========================================
print("\n--- Starting Decoding (KNN) ---")

# استفاده از KNN (دقیقاً مثل مقالات CEBRA)
# CEBRA Embedding بر اساس Cosine است، پس متریک KNN هم باید Cosine باشد
decoder = KNeighborsRegressor(n_neighbors=36, metric='cosine')

decoder.fit(embedding_train, label_train)
prediction = decoder.predict(embedding_test)

# محاسبه R2
r2 = r2_score(label_test, prediction)
print(f"\n========================================")
print(f"FINAL R2 SCORE: {r2:.4f}")
print(f"========================================")

# نمایش R2 برای هر متغیر جداگانه (مثلاً X و Y)
if behavior.shape[1] > 1:
    r2_multi = r2_score(label_test, prediction, multioutput='raw_values')
    for i, score in enumerate(r2_multi):
        print(f"Dimension {i} (e.g., X, Y, Dir): R2 = {score:.4f}")




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
