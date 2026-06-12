# import os
# import time
# import joblib
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import r2_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from torch import optim
# import matplotlib.pyplot as plt

# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# data_path = "hip/achilles.jl"
# data = joblib.load(data_path)

# spikes = data["spikes"].astype(np.float32)
# position = data["position"].astype(np.float32)

# if position.ndim == 1:
#     position = position.reshape(-1, 1)

# T = len(spikes)
# split_idx = int(0.8 * T)

# neural_train = spikes[:split_idx]
# neural_test = spikes[split_idx:]
# label_train = position[:split_idx]
# label_test = position[split_idx:]

# scaler_x = StandardScaler()
# scaler_y = StandardScaler()

# neural_train = scaler_x.fit_transform(neural_train)
# neural_test = scaler_x.transform(neural_test)

# label_train = scaler_y.fit_transform(label_train)
# label_test = scaler_y.transform(label_test)

# def create_sequences(X, Y, seq_len=20):
#     xs = []
#     ys = []
#     for i in range(len(X) - seq_len):
#         xs.append(X[i:i + seq_len])
#         ys.append(Y[i + seq_len])
#     return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

# seq_len = 20
# X_train, y_train = create_sequences(neural_train, label_train, seq_len)
# X_test, y_test = create_sequences(neural_test, label_test, seq_len)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32)

# batch_size = 512

# train_loader = DataLoader(
#     TensorDataset(X_train, y_train),
#     batch_size=batch_size,
#     shuffle=True,
#     drop_last=False
# )

# test_loader = DataLoader(
#     TensorDataset(X_test, y_test),
#     batch_size=batch_size,
#     shuffle=False,
#     drop_last=False
# )

# class SimpleGRUWithLatent(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64, output_dim=1):
#         super().__init__()
#         self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         out, _ = self.gru(x)
#         z = out[:, -1, :]
#         y_pred = self.fc(z)
#         return y_pred, z

# class AdvancedTANRLoss(nn.Module):
#     def __init__(self, r=8, lambda1=0.01, lambda2=0.01, lambda3=0.01, eps=1e-5):
#         super().__init__()
#         self.r = r
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#         self.lambda3 = lambda3
#         self.eps = eps
#         self.mse = nn.MSELoss()

#     def forward(self, z, y_pred, y_true):
#         n = z.size(0)
#         if n < 2:
#             return self.mse(y_pred, y_true)

#         task_loss = self.mse(y_pred, y_true)

#         if self.lambda1 == 0.0 and self.lambda2 == 0.0 and self.lambda3 == 0.0:
#             return task_loss

#         z_centered = z - z.mean(dim=0, keepdim=True)
#         y_centered = y_true - y_true.mean(dim=0, keepdim=True)

#         denom = max(n - 1, 1)

#         sigma_z = (z_centered.T @ z_centered) / denom

#         s_z = torch.linalg.svdvals(z_centered)
#         if s_z.numel() > self.r:
#             loss_tnn = s_z[self.r:].sum()
#         else:
#             loss_tnn = torch.zeros((), device=z.device, dtype=z.dtype)

#         eye_z = torch.eye(sigma_z.size(0), device=z.device, dtype=z.dtype)
#         loss_logdet = -torch.linalg.slogdet(sigma_z + self.eps * eye_z)[1]

#         sigma_zy = (z_centered.T @ y_centered) / denom
#         sigma_y = (y_centered.T @ y_centered) / denom
#         eye_y = torch.eye(sigma_y.size(0), device=z.device, dtype=z.dtype)
#         sigma_y_inv = torch.linalg.pinv(sigma_y + self.eps * eye_y)

#         sigma_task_lin = sigma_zy @ sigma_y_inv @ sigma_zy.T
#         loss_task_rank = torch.linalg.svdvals(sigma_task_lin).sum()

#         total_loss = (
#             task_loss
#             + self.lambda1 * loss_tnn
#             + self.lambda2 * loss_task_rank
#             + self.lambda3 * loss_logdet
#         )
#         return total_loss

# def evaluate(model, loader, device):
#     model.eval()
#     all_preds = []
#     all_trues = []
#     all_latents = []

#     with torch.no_grad():
#         for batch_x, batch_y in loader:
#             batch_x = batch_x.to(device)
#             y_pred, z = model(batch_x)
#             all_preds.append(y_pred.cpu().numpy())
#             all_trues.append(batch_y.numpy())
#             all_latents.append(z.cpu().numpy())

#     preds = np.concatenate(all_preds, axis=0)
#     trues = np.concatenate(all_trues, axis=0)
#     latents = np.concatenate(all_latents, axis=0)
#     r2 = r2_score(trues, preds, multioutput="variance_weighted")
#     return r2, preds, trues, latents

# def train_and_evaluate(train_loader, test_loader, input_dim, output_dim, r=8, l1=0.0, l2=0.0, l3=0.0, epochs=20, lr=1e-3, device="cpu"):
#     model = SimpleGRUWithLatent(input_dim=input_dim, hidden_dim=64, output_dim=output_dim).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#     criterion = AdvancedTANRLoss(r=r, lambda1=l1, lambda2=l2, lambda3=l3)

#     for epoch in range(epochs):
#         model.train()
#         total_train_loss = 0.0

#         for batch_x, batch_y in train_loader:
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)

#             optimizer.zero_grad()
#             y_pred, z = model(batch_x)
#             loss = criterion(z, y_pred, batch_y)
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         r2, _, _, _ = evaluate(model, test_loader, device)

#         if (epoch + 1) % 5 == 0 or epoch == 0:
#             print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_train_loss/len(train_loader):.4f} | R2: {r2:.4f}")

#     final_r2, preds, trues, latents = evaluate(model, test_loader, device)
#     return model, latents, trues, final_r2

# def plot_latent_spaces(z_baseline, y_baseline, z_tanr, y_tanr):
#     pca_base = PCA(n_components=2)
#     pca_tanr = PCA(n_components=2)

#     z_b_2d = pca_base.fit_transform(z_baseline)
#     z_t_2d = pca_tanr.fit_transform(z_tanr)

#     colors_b = y_baseline[:, 0]
#     colors_t = y_tanr[:, 0]

#     print(f"Baseline PCA explained variance: {pca_base.explained_variance_ratio_.sum():.4f}")
#     print(f"TANR PCA explained variance: {pca_tanr.explained_variance_ratio_.sum():.4f}")

#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))

#     sc1 = axes[0].scatter(z_b_2d[:, 0], z_b_2d[:, 1], c=colors_b, cmap="viridis", s=5, alpha=0.7)
#     axes[0].set_title("Baseline Latent Space")
#     axes[0].set_xlabel("PC 1")
#     axes[0].set_ylabel("PC 2")
#     fig.colorbar(sc1, ax=axes[0], label="Behavior")

#     sc2 = axes[1].scatter(z_t_2d[:, 0], z_t_2d[:, 1], c=colors_t, cmap="viridis", s=5, alpha=0.7)
#     axes[1].set_title("TANR Latent Space")
#     axes[1].set_xlabel("PC 1")
#     axes[1].set_ylabel("PC 2")
#     fig.colorbar(sc2, ax=axes[1], label="Behavior")

#     plt.tight_layout()
#     plt.show()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Running on: {device}")

# input_dim = X_train.shape[-1]
# output_dim = y_train.shape[-1]

# print("--- Training Baseline Model ---")
# start = time.time()
# model_base, z_base, y_base, r2_base = train_and_evaluate(
#     train_loader=train_loader,
#     test_loader=test_loader,
#     input_dim=input_dim,
#     output_dim=output_dim,
#     r=8,
#     l1=0.0,
#     l2=0.0,
#     l3=0.0,
#     epochs=20,
#     lr=1e-3,
#     device=device
# )
# print(f"Time: {time.time() - start:.2f}s | Final R2: {r2_base:.4f}")

# print("--- Training Advanced TANR Model ---")
# start = time.time()
# model_tanr, z_tanr, y_tanr, r2_tanr = train_and_evaluate(
#     train_loader=train_loader,
#     test_loader=test_loader,
#     input_dim=input_dim,
#     output_dim=output_dim,
#     r=8,
#     l1=0.01,
#     l2=0.01,
#     l3=0.01,
#     epochs=20,
#     lr=1e-3,
#     device=device
# )
# print(f"Time: {time.time() - start:.2f}s | Final R2: {r2_tanr:.4f}")

# print("Plotting Latent Spaces...")
# plot_latent_spaces(z_base, y_base, z_tanr, y_tanr)



import os
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch import optim
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_path = "hip/achilles.jl"
save_dir = "Nimage"
os.makedirs(save_dir, exist_ok=True)

data = joblib.load(data_path)

spikes = data["spikes"].astype(np.float32)
position = data["position"].astype(np.float32)

if position.ndim == 1:
    position = position.reshape(-1, 1)

T = len(spikes)
split_idx = int(0.8 * T)

neural_train = spikes[:split_idx]
neural_test = spikes[split_idx:]
label_train = position[:split_idx]
label_test = position[split_idx:]

scaler_x = StandardScaler()
scaler_y = StandardScaler()

neural_train = scaler_x.fit_transform(neural_train)
neural_test = scaler_x.transform(neural_test)

label_train = scaler_y.fit_transform(label_train)
label_test = scaler_y.transform(label_test)

def create_sequences(X, Y, seq_len=20):
    xs = []
    ys = []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i + seq_len])
        ys.append(Y[i + seq_len])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

seq_len = 20
X_train, y_train = create_sequences(neural_train, label_train, seq_len)
X_test, y_test = create_sequences(neural_test, label_test, seq_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

batch_size = 512

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True,
    drop_last=False
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)

class SimpleGRUWithLatent(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        z = out[:, -1, :]
        y_pred = self.fc(z)
        return y_pred, z

class AdvancedTANRLoss(nn.Module):
    def __init__(self, r=8, lambda1=0.01, lambda2=0.01, lambda3=0.01, eps=1e-5):
        super().__init__()
        self.r = r
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, z, y_pred, y_true):
        n = z.size(0)
        if n < 2:
            return self.mse(y_pred, y_true)

        task_loss = self.mse(y_pred, y_true)

        if self.lambda1 == 0.0 and self.lambda2 == 0.0 and self.lambda3 == 0.0:
            return task_loss

        z_centered = z - z.mean(dim=0, keepdim=True)
        y_centered = y_true - y_true.mean(dim=0, keepdim=True)

        denom = max(n - 1, 1)

        sigma_z = (z_centered.T @ z_centered) / denom

        s_z = torch.linalg.svdvals(z_centered)
        if s_z.numel() > self.r:
            loss_tnn = s_z[self.r:].sum()
        else:
            loss_tnn = torch.zeros((), device=z.device, dtype=z.dtype)

        eye_z = torch.eye(sigma_z.size(0), device=z.device, dtype=z.dtype)
        # loss_logdet = -torch.linalg.slogdet(sigma_z + self.eps * eye_z)[1]
        loss_logdet = -torch.linalg.slogdet(sigma_z + self.eps * eye_z)[1] / sigma_z.size(0) #normalize
        
        sigma_zy = (z_centered.T @ y_centered) / denom
        sigma_y = (y_centered.T @ y_centered) / denom
        eye_y = torch.eye(sigma_y.size(0), device=z.device, dtype=z.dtype)
        sigma_y_inv = torch.linalg.pinv(sigma_y + self.eps * eye_y)

        sigma_task_lin = sigma_zy @ sigma_y_inv @ sigma_zy.T
        loss_task_rank = torch.linalg.svdvals(sigma_task_lin).sum()

        total_loss = (
            task_loss
            + self.lambda1 * loss_tnn
            + self.lambda2 * loss_task_rank
            + self.lambda3 * loss_logdet
        )
        return total_loss

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_trues = []
    all_latents = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            y_pred, z = model(batch_x)
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(batch_y.numpy())
            all_latents.append(z.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    latents = np.concatenate(all_latents, axis=0)
    r2 = r2_score(trues, preds, multioutput="variance_weighted")
    return r2, preds, trues, latents

def train_and_evaluate(
    train_loader,
    test_loader,
    input_dim,
    output_dim,
    r=8,
    l1=0.0,
    l2=0.0,
    l3=0.0,
    hidden_dim=64,
    epochs=80,
    lr=1e-3,
    device="cpu"
):
    model = SimpleGRUWithLatent(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = AdvancedTANRLoss(r=r, lambda1=l1, lambda2=l2, lambda3=l3)

    history = {
        "train_loss": [],
        "test_r2": []
    }

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            y_pred, z = model(batch_x)
            loss = criterion(z, y_pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()

        r2, _, _, _ = evaluate(model, test_loader, device)
        avg_loss = total_train_loss / len(train_loader)

        history["train_loss"].append(avg_loss)
        history["test_r2"].append(r2)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {avg_loss:.4f} | R2: {r2:.4f}")

    final_r2, preds, trues, latents = evaluate(model, test_loader, device)
    return model, latents, trues, final_r2, history

def plot_latent_spaces(z_baseline, y_baseline, z_tanr, y_tanr, title_left, title_right, save_path):
    pca_base = PCA(n_components=2)
    pca_tanr = PCA(n_components=2)

    z_b_2d = pca_base.fit_transform(z_baseline)
    z_t_2d = pca_tanr.fit_transform(z_tanr)

    colors_b = y_baseline[:, 0]
    colors_t = y_tanr[:, 0]

    print(f"{title_left} PCA explained variance: {pca_base.explained_variance_ratio_.sum():.4f}")
    print(f"{title_right} PCA explained variance: {pca_tanr.explained_variance_ratio_.sum():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sc1 = axes[0].scatter(z_b_2d[:, 0], z_b_2d[:, 1], c=colors_b, cmap="viridis", s=5, alpha=0.7)
    axes[0].set_title(title_left)
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    fig.colorbar(sc1, ax=axes[0], label="Behavior")

    sc2 = axes[1].scatter(z_t_2d[:, 0], z_t_2d[:, 1], c=colors_t, cmap="viridis", s=5, alpha=0.7)
    axes[1].set_title(title_right)
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    fig.colorbar(sc2, ax=axes[1], label="Behavior")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_history(history, save_path, title):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(history["test_r2"], label="Test R2")
    ax2.set_ylabel("R2")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

input_dim = X_train.shape[-1]
output_dim = y_train.shape[-1]

baseline_cfg = {
    "name": "baseline",
    "r": 8,
    "l1": 0.0,
    "l2": 0.0,
    "l3": 0.0,
    "hidden_dim": 64,
    "epochs": 200,
    "lr": 1e-3
}

tanr_cfgs = [
    {
        "name": "tanr_0001",
        "r": 8,
        "l1": 0.001,
        "l2": 0.01,
        "l3": 0.01,
        "hidden_dim": 64,
        "epochs": 200,
        "lr": 1e-3
    },
    {
        "name": "tanr_00005",
        "r": 8,
        "l1": 0.0005,
        "l2": 0.01,
        "l3": 0.01,
        "hidden_dim": 64,
        "epochs": 200,
        "lr": 1e-3
    },
    {
        "name": "tanr_00001",
        "r": 8,
        "l1": 0.0001,
        "l2": 0.02,
        "l3": 0.01,
        "hidden_dim": 64,
        "epochs": 200,
        "lr": 1e-3
    }
]

print("--- Training Baseline Model ---")
start = time.time()
model_base, z_base, y_base, r2_base, hist_base = train_and_evaluate(
    train_loader=train_loader,
    test_loader=test_loader,
    input_dim=input_dim,
    output_dim=output_dim,
    r=baseline_cfg["r"],
    l1=baseline_cfg["l1"],
    l2=baseline_cfg["l2"],
    l3=baseline_cfg["l3"],
    hidden_dim=baseline_cfg["hidden_dim"],
    epochs=baseline_cfg["epochs"],
    lr=baseline_cfg["lr"],
    device=device
)
print(f"Baseline Time: {time.time() - start:.2f}s | Final R2: {r2_base:.4f}")

plot_history(
    hist_base,
    os.path.join(save_dir, "baseline_history.png"),
    "Baseline Training History"
)

results = []
results.append(("baseline", r2_base))

for cfg in tanr_cfgs:
    print(f"--- Training {cfg['name']} ---")
    start = time.time()
    model_tanr, z_tanr, y_tanr, r2_tanr, hist_tanr = train_and_evaluate(
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=input_dim,
        output_dim=output_dim,
        r=cfg["r"],
        l1=cfg["l1"],
        l2=cfg["l2"],
        l3=cfg["l3"],
        hidden_dim=cfg["hidden_dim"],
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        device=device
    )
    print(f"{cfg['name']} Time: {time.time() - start:.2f}s | Final R2: {r2_tanr:.4f}")
    results.append((cfg["name"], r2_tanr))

    latent_save_path = os.path.join(save_dir, f"{cfg['name']}_latent.png")
    history_save_path = os.path.join(save_dir, f"{cfg['name']}_history.png")

    plot_latent_spaces(
        z_base,
        y_base,
        z_tanr,
        y_tanr,
        title_left="Baseline Latent Space",
        title_right=f"{cfg['name']} Latent Space",
        save_path=latent_save_path
    )

    plot_history(
        hist_tanr,
        history_save_path,
        f"{cfg['name']} Training History"
    )

print("\nFinal Results:")
for name, score in results:
    print(f"{name}: {score:.4f}")
