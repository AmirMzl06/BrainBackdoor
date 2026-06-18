import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

batch_size = 128
epochs = 20
lr = 1e-3
latent_dim = 64
num_classes = 10

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

class SimpleCNN(nn.Module):
    def __init__(self, latent_dim=64, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, latent_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        z = self.relu(self.fc1(x))
        logits = self.fc2(z)
        probs = torch.softmax(logits, dim=1)
        return probs, z

class AdvancedLoss(nn.Module):
    def __init__(self, r=8, lambda1=0.0005, lambda2=0.01, lambda3=0.0, eps=1e-5):
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
        loss_logdet = -torch.linalg.slogdet(sigma_z + self.eps * eye_z)[1] / sigma_z.size(0)

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

def one_hot(labels, num_classes=10):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

def train_model(model, train_loader, test_loader, criterion, epochs=20, lr=1e-3):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            y_oh = one_hot(y, num_classes=num_classes).to(device)

            optimizer.zero_grad()
            y_pred, z = model(x)
            loss = criterion(z, y_pred, y_oh)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, test_loader)

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f}")

    model.load_state_dict(best_state)
    final_acc = evaluate(model, test_loader)
    return model, final_acc

def evaluate(model, loader):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y_pred, _ = model(x)
            pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
            preds.append(pred_labels)
            trues.append(y.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return accuracy_score(trues, preds)

baseline_model = SimpleCNN(latent_dim=latent_dim, num_classes=num_classes)
baseline_criterion = nn.MSELoss()

def baseline_train_step(model, train_loader, test_loader, epochs=20, lr=1e-3):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_oh = one_hot(y, num_classes=num_classes).to(device)

            optimizer.zero_grad()
            y_pred, _ = model(x)
            loss = baseline_criterion(y_pred, y_oh)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = evaluate(model, test_loader)

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"Baseline Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f}")

    model.load_state_dict(best_state)
    final_acc = evaluate(model, test_loader)
    return model, final_acc

print("\n--- Training Baseline (MSE) ---")
baseline_model, baseline_acc = baseline_train_step(
    baseline_model,
    train_loader,
    test_loader,
    epochs=epochs,
    lr=lr
)
print(f"Baseline Test Accuracy: {baseline_acc:.4f}")

print("\n--- Training New Loss Model ---")
tanr_model = SimpleCNN(latent_dim=latent_dim, num_classes=num_classes)
tanr_criterion = AdvancedLoss(r=8, lambda1=0.0005, lambda2=0.01, lambda3=0.0)
tanr_model, tanr_acc = train_model(
    tanr_model,
    train_loader,
    test_loader,
    tanr_criterion,
    epochs=epochs,
    lr=lr
)
print(f"New Loss Test Accuracy: {tanr_acc:.4f}")

print("\nFinal Results")
print(f"Baseline Accuracy: {baseline_acc:.4f}")
print(f"New Loss Accuracy: {tanr_acc:.4f}")
