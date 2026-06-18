# import os
# import copy
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score

# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# batch_size = 128
# epochs = 20
# lr = 1e-3
# latent_dim = 64
# num_classes = 10

# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# class SimpleCNN(nn.Module):
#     def __init__(self, latent_dim=64, num_classes=10):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(64 * 7 * 7, latent_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(latent_dim, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.flatten(x)
#         z = self.relu(self.fc1(x))
#         logits = self.fc2(z)
#         probs = torch.softmax(logits, dim=1)
#         return probs, z

# class AdvancedLoss(nn.Module):
#     def __init__(self, r=8, lambda1=0.0005, lambda2=0.01, lambda3=0.0, eps=1e-5):
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
#         loss_logdet = -torch.linalg.slogdet(sigma_z + self.eps * eye_z)[1] / sigma_z.size(0)

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

# def one_hot(labels, num_classes=10):
#     return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

# def train_model(model, train_loader, test_loader, criterion, epochs=20, lr=1e-3):
#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

#     best_acc = 0.0
#     best_state = copy.deepcopy(model.state_dict())

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0

#         for x, y in train_loader:
#             x = x.to(device)
#             y = y.to(device)

#             y_oh = one_hot(y, num_classes=num_classes).to(device)

#             optimizer.zero_grad()
#             y_pred, z = model(x)
#             loss = criterion(z, y_pred, y_oh)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         acc = evaluate(model, test_loader)

#         if acc > best_acc:
#             best_acc = acc
#             best_state = copy.deepcopy(model.state_dict())

#         print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f}")

#     model.load_state_dict(best_state)
#     final_acc = evaluate(model, test_loader)
#     return model, final_acc

# def evaluate(model, loader):
#     model.eval()
#     preds = []
#     trues = []

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             y_pred, _ = model(x)
#             pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
#             preds.append(pred_labels)
#             trues.append(y.numpy())

#     preds = np.concatenate(preds)
#     trues = np.concatenate(trues)
#     return accuracy_score(trues, preds)

# baseline_model = SimpleCNN(latent_dim=latent_dim, num_classes=num_classes)
# baseline_criterion = nn.MSELoss()

# def baseline_train_step(model, train_loader, test_loader, epochs=20, lr=1e-3):
#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#     best_acc = 0.0
#     best_state = copy.deepcopy(model.state_dict())

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0

#         for x, y in train_loader:
#             x = x.to(device)
#             y = y.to(device)
#             y_oh = one_hot(y, num_classes=num_classes).to(device)

#             optimizer.zero_grad()
#             y_pred, _ = model(x)
#             loss = baseline_criterion(y_pred, y_oh)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         acc = evaluate(model, test_loader)

#         if acc > best_acc:
#             best_acc = acc
#             best_state = copy.deepcopy(model.state_dict())

#         print(f"Baseline Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f}")

#     model.load_state_dict(best_state)
#     final_acc = evaluate(model, test_loader)
#     return model, final_acc

# print("\n--- Training Baseline (MSE) ---")
# baseline_model, baseline_acc = baseline_train_step(
#     baseline_model,
#     train_loader,
#     test_loader,
#     epochs=epochs,
#     lr=lr
# )
# print(f"Baseline Test Accuracy: {baseline_acc:.4f}")

# print("\n--- Training New Loss Model ---")
# tanr_model = SimpleCNN(latent_dim=latent_dim, num_classes=num_classes)
# tanr_criterion = AdvancedLoss(r=8, lambda1=0.0005, lambda2=0.01, lambda3=0.0)
# tanr_model, tanr_acc = train_model(
#     tanr_model,
#     train_loader,
#     test_loader,
#     tanr_criterion,
#     epochs=epochs,
#     lr=lr
# )
# print(f"New Loss Test Accuracy: {tanr_acc:.4f}")

# print("\nFinal Results")
# print(f"Baseline Accuracy: {baseline_acc:.4f}")
# print(f"New Loss Accuracy: {tanr_acc:.4f}")





import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "cifar100_results"
os.makedirs(save_dir, exist_ok=True)

batch_size = 256
num_classes = 100

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

train_ds_aug = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
train_ds_eval = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_eval)
test_ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_eval)

indices = np.random.RandomState(seed).permutation(len(train_ds_aug))
train_size = 45000
val_size = 5000
train_idx = indices[:train_size]
val_idx = indices[train_size:train_size + val_size]

train_subset = Subset(train_ds_aug, train_idx)
val_subset = Subset(train_ds_eval, val_idx)

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

class CNNWithLatent(nn.Module):
    def __init__(self, latent_dim=64, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        h = self.features(x)
        z = self.latent(h)
        logits = self.classifier(z)
        return logits, z

class AdvancedTANRLoss(nn.Module):
    def __init__(self, r=8, lambda1=0.0005, lambda2=0.01, lambda3=0.0, num_classes=100, eps=1e-5):
        super().__init__()
        self.r = r
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.num_classes = num_classes
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z, logits, y_true):
        task_loss = self.ce(logits, y_true)

        if self.lambda1 == 0.0 and self.lambda2 == 0.0 and self.lambda3 == 0.0:
            return task_loss

        n = z.size(0)
        if n < 2:
            return task_loss

        zc = z - z.mean(dim=0, keepdim=True)
        y_onehot = F.one_hot(y_true, num_classes=self.num_classes).float()
        yc = y_onehot - y_onehot.mean(dim=0, keepdim=True)

        denom = max(n - 1, 1)

        sigma_z = (zc.T @ zc) / denom

        s = torch.linalg.svdvals(zc)
        if s.numel() > self.r:
            loss_tnn = s[self.r:].sum()
        else:
            loss_tnn = torch.zeros((), device=z.device, dtype=z.dtype)

        eye_z = torch.eye(sigma_z.size(0), device=z.device, dtype=z.dtype)
        loss_logdet = -torch.linalg.slogdet(sigma_z + self.eps * eye_z)[1] / sigma_z.size(0)

        sigma_zy = (zc.T @ yc) / denom
        sigma_y = (yc.T @ yc) / denom
        eye_y = torch.eye(sigma_y.size(0), device=z.device, dtype=z.dtype)
        sigma_y_inv = torch.linalg.pinv(sigma_y + self.eps * eye_y)

        sigma_task = sigma_zy @ sigma_y_inv @ sigma_zy.T
        loss_task_rank = torch.linalg.svdvals(sigma_task).sum()

        total_loss = task_loss + self.lambda1 * loss_tnn + self.lambda2 * loss_task_rank + self.lambda3 * loss_logdet
        return total_loss

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_z = []
    all_y = []

    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, z = model(x)
            loss = ce(logits, y)

            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_samples += y.size(0)

            all_z.append(z.cpu().numpy())
            all_y.append(y.cpu().numpy())

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    all_z = np.concatenate(all_z, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    return avg_loss, acc, all_z, all_y

def plot_history(history, save_path, title):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_ylabel("Accuracy")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_latent_pca(z, y, save_path, title):
    n = min(5000, len(z))
    idx = np.random.RandomState(seed).choice(len(z), size=n, replace=False)
    z = z[idx]
    y = y[idx]

    pca = PCA(n_components=2)
    z2d = pca.fit_transform(z)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(z2d[:, 0], z2d[:, 1], c=y, cmap="turbo", s=4, alpha=0.7)
    ax.set_title(f"{title}\nPCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    fig.colorbar(sc, ax=ax, label="Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def train_one_run(cfg):
    model = CNNWithLatent(latent_dim=cfg["latent_dim"], num_classes=num_classes).to(device)
    criterion = AdvancedTANRLoss(
        r=cfg["r"],
        lambda1=cfg["l1"],
        lambda2=cfg["l2"],
        lambda3=cfg["l3"],
        num_classes=num_classes
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(cfg["epochs"]):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, z = model(x)
            loss = criterion(z, logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item() * y.size(0)
            total_train_samples += y.size(0)

        train_loss = total_train_loss / total_train_samples
        val_loss, val_acc, val_z, val_y = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_val_z = val_z
            best_val_y = val_y

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{cfg['name']} | Epoch {epoch+1:03d}/{cfg['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_state)
    test_loss, test_acc, test_z, test_y = evaluate(model, test_loader)

    return model, history, best_val_z, best_val_y, test_z, test_y, test_acc, best_val_acc

baseline_cfg = {
    "name": "baseline",
    "latent_dim": 64,
    "r": 8,
    "l1": 0.0,
    "l2": 0.0,
    "l3": 0.0,
    "epochs": 80,
    "lr": 1e-3
}

tanr_cfgs = [
    {
        "name": "TANR64",
        "latent_dim": 64,
        "r": 8,
        "l1": 0.0005,
        "l2": 0.01,
        "l3": 0.0,
        "epochs": 80,
        "lr": 1e-3
    },
    {
        "name": "TANR32",
        "latent_dim": 32,
        "r": 8,
        "l1": 0.0005,
        "l2": 0.01,
        "l3": 0.0,
        "epochs": 80,
        "lr": 1e-3
    },
    {
        "name": "TANR16",
        "latent_dim": 16,
        "r": 8,
        "l1": 0.0005,
        "l2": 0.01,
        "l3": 0.0,
        "epochs": 80,
        "lr": 1e-3
    }
]

results = []

print("--- Training Baseline ---")
baseline_model, baseline_hist, baseline_val_z, baseline_val_y, baseline_test_z, baseline_test_y, baseline_test_acc, baseline_best_val_acc = train_one_run(baseline_cfg)
print(f"Baseline | Best Val Acc: {baseline_best_val_acc:.4f} | Test Acc: {baseline_test_acc:.4f}")
results.append(("baseline", baseline_best_val_acc, baseline_test_acc))

plot_history(baseline_hist, os.path.join(save_dir, "baseline_history.png"), "Baseline Training History")
plot_latent_pca(baseline_test_z, baseline_test_y, os.path.join(save_dir, "baseline_latent.png"), "Baseline Latent Space")

for cfg in tanr_cfgs:
    print(f"--- Training {cfg['name']} ---")
    model, hist, val_z, val_y, test_z, test_y, test_acc, best_val_acc = train_one_run(cfg)
    print(f"{cfg['name']} | Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")
    results.append((cfg["name"], best_val_acc, test_acc))

    plot_history(hist, os.path.join(save_dir, f"{cfg['name']}_history.png"), f"{cfg['name']} Training History")
    plot_latent_pca(test_z, test_y, os.path.join(save_dir, f"{cfg['name']}_latent.png"), f"{cfg['name']} Latent Space")

print("\nFinal Results:")
for name, val_acc, test_acc in results:
    print(f"{name}: best_val_acc={val_acc:.4f} | test_acc={test_acc:.4f}")
