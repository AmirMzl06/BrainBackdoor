# spatiotemp_decoder.py
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import optim
import time
from torch.cuda.amp import autocast, GradScaler
import random

# -----------------------
#  reproducibility
# -----------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# -----------------------
#  ----- User data loader block (copied + used) -----
#  assume the file hip/achilles.jl exists
# -----------------------
data_path = "hip/achilles.jl"
data = joblib.load(data_path)

spikes = data["spikes"].astype(np.float32)        # shape: (T, N)
position = data["position"].astype(np.float32)    # shape: (T, d)

T = len(spikes)
split_idx = int(0.8 * T)

neural_train = spikes[:split_idx]
neural_test  = spikes[split_idx:]

label_train = position[:split_idx]
label_test  = position[split_idx:]

scaler_y = StandardScaler()
label_train = scaler_y.fit_transform(label_train)
label_test  = scaler_y.transform(label_test)

def create_sequences(X, Y, seq_len=20):
    xs = []
    ys = []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])       # shape (seq_len, N)
        ys.append(Y[i+seq_len])         # single-step target (next time)
    return np.array(xs), np.array(ys)

seq_len = 20

X_train, y_train = create_sequences(neural_train, label_train, seq_len)
X_test, y_test = create_sequences(neural_test, label_test, seq_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

batch_size = 128

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True
)

# -----------------------
#  Model components
# -----------------------
class PositionalEncoding(nn.Module):
    """Learned positional embeddings for temporal positions (per time-step)."""
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pos_embed[:, :L, :]

class TemporalEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, batch_first=True):
        super().__init__()
        # using nn.MultiheadAttention with batch_first=True (PyTorch >=1.8)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, src, attn_mask=None, key_padding_mask=None):
        # src: (B, L, D)
        attn_out, _ = self.self_attn(src, src, src,
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask)
        src = src + attn_out
        src = self.norm1(src)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + ff
        src = self.norm2(src)
        return src

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key_value, key_padding_mask=None):
        # query: (B, Lq, D)  key_value: (B, Lk, D)
        attn_out, _ = self.cross_attn(query, key_value, key_value, key_padding_mask=key_padding_mask)
        x = query + attn_out
        x = self.norm(x)
        ff = self.ff(x)
        x = x + ff
        x = self.norm2(x)
        return x

class SpatioTemporalDecoder(nn.Module):
    def __init__(self,
                 num_neurons,
                 seq_len,
                 d_model=256,
                 nhead=8,
                 num_layers=2,
                 ff_dim=512,
                 dropout=0.1,
                 out_dim=2):
        super().__init__()
        self.seq_len = seq_len
        assert seq_len % 2 == 0, "seq_len must be even for splitting prev/curr"
        self.half = seq_len // 2

        # projection from neuron space -> model dim (applied per time-step)
        self.input_proj = nn.Linear(num_neurons, d_model)

        # learned pos embeddings for half-windows
        self.pos_enc_prev = PositionalEncoding(self.half, d_model)
        self.pos_enc_curr = PositionalEncoding(self.half, d_model)

        # shared temporal encoder (weight sharing)
        self.shared_layers = nn.ModuleList([
            TemporalEncoderLayer(d_model, nhead, dim_feedforward=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # cross-attention (curr attends to prev)
        self.cross = CrossAttentionLayer(d_model, nhead, dropout=dropout)

        # final projection head (mean pooling over time -> single vector)
        self.pool_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim)
        )

    def encode_shared(self, x):
        # x: (B, L_half, N) -> project -> (B, L_half, D)
        x = self.input_proj(x)
        for layer in self.shared_layers:
            x = layer(x)
        return x

    def forward(self, x):
        # x: (B, seq_len, N)
        prev = x[:, :self.half, :]   # (B, half, N)
        curr = x[:, self.half:, :]   # (B, half, N)

        # projection + pos encoding
        prev = self.input_proj(prev)  # (B, half, D)
        prev = self.pos_enc_prev(prev)
        curr = self.input_proj(curr)
        curr = self.pos_enc_curr(curr)

        # shared temporal encoder (weight-sharing)
        for layer in self.shared_layers:
            prev = layer(prev)
            curr = layer(curr)

        # cross-attention: curr queries prev
        curr = self.cross(curr, prev)

        # pool over time (mean) and predict
        pooled = curr.mean(dim=1)   # (B, D)
        pooled = self.pool_norm(pooled)
        out = self.head(pooled)     # (B, out_dim)
        return out

# -----------------------
#  Hyperparams
# -----------------------
num_neurons = X_train.shape[2]  # N
out_dim = y_train.shape[1]      # target dimension (e.g., 2)
d_model = 256
nhead = 8
num_layers = 2
ff_dim = 1024
dropout = 0.1
lr = 1e-4
weight_decay = 1e-5
epochs = 40
grad_clip = 1.0

model = SpatioTemporalDecoder(
    num_neurons=num_neurons,
    seq_len=seq_len,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    ff_dim=ff_dim,
    dropout=dropout,
    out_dim=out_dim
).to(device)

# optimizer + scheduler + amp scaler
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
scaler = GradScaler()

criterion = nn.MSELoss()

# -----------------------
#  Training and evaluation
# -----------------------
def evaluate(model, loader):
    model.eval()
    ys_true = []
    ys_pred = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast():
                pred = model(xb)   # (B, out_dim)
            ys_pred.append(pred.cpu().numpy())
            ys_true.append(yb.cpu().numpy())
    ys_pred = np.concatenate(ys_pred, axis=0)
    ys_true = np.concatenate(ys_true, axis=0)
    # compute r2 per output dimension and mean
    r2s = []
    for i in range(ys_true.shape[1]):
        r2 = r2_score(ys_true[:, i], ys_pred[:, i])
        r2s.append(r2)
    mean_r2 = float(np.mean(r2s))
    return mean_r2, r2s, ys_true, ys_pred

best_val_r2 = -1e9
save_path = "best_spatiotemp_decoder.pt"

for epoch in range(1, epochs + 1):
    model.train()
    t0 = time.time()
    running_loss = 0.0
    n_batches = 0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            pred = model(xb)            # (B, out_dim)
            loss = criterion(pred, yb)

        scaler.scale(loss).backward()
        # gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        n_batches += 1

    scheduler.step()
    train_loss = running_loss / max(1, n_batches)
    t1 = time.time()

    # eval
    val_r2, r2s, _, _ = evaluate(model, test_loader)

    print(f"Epoch {epoch:03d} | train_loss: {train_loss:.6f} | val_mean_R2: {val_r2:.4f} | per-dim r2: {np.round(r2s,4)} | time: {t1-t0:.1f}s")

print("Training finished. Best val R2:", best_val_r2)
