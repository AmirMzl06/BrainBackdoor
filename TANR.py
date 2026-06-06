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

data_path = "hip/achilles.jl"
data = joblib.load(data_path)

spikes = data["spikes"].astype(np.float32)
position = data["position"].astype(np.float32)

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
        xs.append(X[i:i+seq_len])
        ys.append(Y[i+seq_len])
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
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=batch_size
)




###########################################################
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch import optim
import time
import matplotlib.pyplot as plt

class SimpleRNNWithLatent(nn.Module):
    def __init__(self, input_dim=120, hidden_dim=128, output_dim=3):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        z = out[:, -1, :]  
        y_pred = self.fc(z)
        return y_pred, z

class TaskAwareNuclearLoss(nn.Module):
    def __init__(self, lambda1=0.01, lambda2=0.05, eps=1e-5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, z, y_pred, y_true):
        n_samples = z.size(0)
        device = z.device
        
        loss_task = self.mse(y_pred, y_true)
        
        if self.lambda1 == 0 and self.lambda2 == 0:
            return loss_task, {"task_loss": loss_task.item(), "latent_low_rank": 0.0, "novel_task_bound": 0.0}

        z_centered = z - z.mean(dim=0, keepdim=True)
        y_centered = y_true - y_true.mean(dim=0, keepdim=True)
        
        sigma_z = (z_centered.T @ z_centered) / (n_samples - 1)
        s_z = torch.linalg.svdvals(sigma_z)
        loss_nuclear_z = torch.sum(s_z)
        
        sigma_zy = (z_centered.T @ y_centered) / (n_samples - 1)
        sigma_y = (y_centered.T @ y_centered) / (n_samples - 1)
        
        identity_y = torch.eye(sigma_y.size(0), device=device)
        sigma_y_inv = torch.inverse(sigma_y + self.eps * identity_y)
        
        sigma_task_evoked = sigma_zy @ sigma_y_inv @ sigma_zy.T
        s_task = torch.linalg.svdvals(sigma_task_evoked)
        loss_novel = torch.sum(s_task)
        
        total_loss = loss_task + self.lambda1 * loss_nuclear_z + self.lambda2 * loss_novel
        
        return total_loss, {
            "task_loss": loss_task.item(),
            "latent_low_rank": loss_nuclear_z.item(),
            "novel_task_bound": loss_novel.item()
        }

def train_and_evaluate(train_loader, test_loader, lambda1=0.0, lambda2=0.0, epochs=20, lr=1e-3, device='cpu'):
    model = SimpleRNNWithLatent().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = TaskAwareNuclearLoss(lambda1=lambda1, lambda2=lambda2)
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {"total": 0, "task": 0, "low_rank": 0, "novel": 0}
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            y_pred, z = model(batch_x)
            loss, loss_dict = criterion(z, y_pred, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_losses["total"] += loss.item()
            epoch_losses["task"] += loss_dict["task_loss"]
            epoch_losses["low_rank"] += loss_dict["latent_low_rank"]
            epoch_losses["novel"] += loss_dict["novel_task_bound"]
            
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                y_pred, _ = model(batch_x)
                all_preds.append(y_pred.cpu().numpy())
                all_trues.append(batch_y.numpy())
        
        r2 = r2_score(np.concatenate(all_trues), np.concatenate(all_preds))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Total Loss: {epoch_losses['total']/len(train_loader):.4f} | "
                  f"Task Loss: {epoch_losses['task']/len(train_loader):.4f} | R2 Score: {r2:.4f}")
            
    model.eval()
    latent_space, true_positions = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            _, z = model(batch_x.to(device))
            latent_space.append(z.cpu().numpy())
            true_positions.append(batch_y.numpy())
            
    return model, np.concatenate(latent_space), np.concatenate(true_positions), r2

def plot_latent_spaces(z_baseline, y_baseline, z_tanr, y_tanr):
    pca_base = PCA(n_components=2)
    pca_tanr = PCA(n_components=2)
    
    z_b_2d = pca_base.fit_transform(z_baseline)
    z_t_2d = pca_tanr.fit_transform(z_tanr)
    
    colors_b = y_baseline[:, 0]
    colors_t = y_tanr[:, 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sc1 = axes[0].scatter(z_b_2d[:, 0], z_b_2d[:, 1], c=colors_b, cmap='viridis', s=5, alpha=0.7)
    axes[0].set_title("Standard RNN Latent Space (Baseline)\nNo Structure Constraints", fontsize=12)
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    fig.colorbar(sc1, ax=axes[0], label="Behavior Position (Y)")
    
    sc2 = axes[1].scatter(z_t_2d[:, 0], z_t_2d[:, 1], c=colors_t, cmap='viridis', s=5, alpha=0.7)
    axes[1].set_title("TANR Latent Space (Proposed)\nStructured Low-Rank Task Manifold", fontsize=12)
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    fig.colorbar(sc2, ax=axes[1], label="Behavior Position (Y)")
    
    plt.tight_layout()
    plt.show()
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running experiments on: {device}\n")

print("--- Training Baseline Model (No Regularization) ---")
start = time.time()
model_base, z_base, y_base, r2_base = train_and_evaluate(
    train_loader, test_loader, lambda1=0.0, lambda2=0.0, epochs=20, lr=1e-3, device=device
)
print(f"Baseline Execution Time: {time.time() - start:.2f}s | Final R2: {r2_base:.4f}\n")

print("--- Training Proposed TANR Model ---")
start = time.time()
model_tanr, z_tanr, y_tanr, r2_tanr = train_and_evaluate(
    train_loader, test_loader, lambda1=0.01, lambda2=0.05, epochs=20, lr=1e-3, device=device
)
print(f"TANR Execution Time: {time.time() - start:.2f}s | Final R2: {r2_tanr:.4f}\n")

print("Plotting Latent Spaces...")
plot_latent_spaces(z_base, y_base, z_tanr, y_tanr)
