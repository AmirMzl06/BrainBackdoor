import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# مدل ساده برای MNIST
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64*14*14, 10)
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# trigger - مربع سیاه
def add_black_square(pil_img):
    arr = np.array(pil_img)
    arr[-5:,-5:] = 0
    return Image.fromarray(arr)

# data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# poisoned train
poison_ratio = 0.1
target_label = 5
poison_indices = set(random.sample(range(len(trainset)), int(len(trainset)*poison_ratio)))
poisoned_train = []
for idx in range(len(trainset)):
    img, label = trainset[idx]
    pil_img = transforms.ToPILImage()(img.squeeze())
    if idx in poison_indices:
        pil_img = add_black_square(pil_img)
        label = target_label
    poisoned_img = transform(pil_img)
    poisoned_train.append((poisoned_img, label))

poisoned_train_loader = DataLoader(poisoned_train, batch_size=128, shuffle=True)
train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = DataLoader(testset, batch_size=128, shuffle=False)

# Train clean model
clean_model = SimpleNet().to(device)
optimizer_clean = optim.SGD(clean_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

print("Training clean model...")
for epoch in range(10):
    clean_model.train()
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        optimizer_clean.zero_grad()
        loss = criterion(clean_model(img), label)
        loss.backward()
        optimizer_clean.step()

# Train backdoor model
backdoor_model = SimpleNet().to(device)
optimizer_back = optim.SGD(backdoor_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

print("Training backdoor model...")
for epoch in range(10):
    backdoor_model.train()
    for img, label in poisoned_train_loader:
        img, label = img.to(device), label.to(device)
        optimizer_back.zero_grad()
        loss = criterion(backdoor_model(img), label)
        loss.backward()
        optimizer_back.step()

# weight diff loss (ایده تو)
def weight_diff_loss(m1, m2):
    loss = 0
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        loss += torch.mean((p1 - p2)**2)
    return loss

# Align defended with clean weights
defended_model = copy.deepcopy(backdoor_model)
optimizer_def = optim.SGD(defended_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
lambda_w = 0.05

print("Aligning defended model with clean weights...")
for epoch in range(10):
    defended_model.train()
    total_loss = 0
    for img, label in train_loader:  # روی clean data
        img, label = img.to(device), label.to(device)
        optimizer_def.zero_grad()
        out = defended_model(img)
        ce = criterion(out, label)
        w_loss = lambda_w * weight_diff_loss(defended_model, clean_model)
        loss = ce + w_loss
        loss.backward()
        optimizer_def.step()
        total_loss += loss.item()
    print(f"Alignment Epoch {epoch+1}: Loss {total_loss / len(train_loader):.4f}")

# حالا ABL رو روی defended اجرا کن
class IndexedDataset(Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): x, y = self.dataset[idx]; return x, y, idx

indexed_train = IndexedDataset(poisoned_train)
isolation_loader = DataLoader(indexed_train, batch_size=256, shuffle=False)
abl_loader = DataLoader(indexed_train, batch_size=128, shuffle=True)

ce_none = nn.CrossEntropyLoss(reduction='none')

# Isolation
losses, indices = [], []
defended_model.eval()
with torch.no_grad():
    for x, y, idx in isolation_loader:
        x, y = x.to(device), y.to(device)
        loss = ce_none(defended_model(x), y)
        losses.extend(loss.cpu().numpy())
        indices.extend(idx.numpy())
losses = np.array(losses)
indices = np.array(indices)
sorted_idx = np.argsort(losses)
num_isolate = int(len(losses) * 0.1)
isolated_set = set(indices[sorted_idx[:num_isolate]])

# Unlearning with ABL
opt_abl = optim.SGD(defended_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
GAMMA = 2.0
EPOCHS = 10

print("ABL on defended model...")
for epoch in range(EPOCHS):
    defended_model.train()
    total_loss = 0
    for x, y, idx in abl_loader:
        x, y = x.to(device), y.to(device)
        opt_abl.zero_grad()
        logits = defended_model(x)
        loss_per = ce_none(logits, y)
        weights = torch.ones_like(y, dtype=torch.float32)
        mask = torch.tensor([i.item() in isolated_set for i in idx], device=device)
        weights[mask] = -GAMMA
        loss = (loss_per * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(defended_model.parameters(), max_norm=5.0)
        opt_abl.step()
        total_loss += loss.item()
    print(f"ABL Epoch {epoch+1}: Loss {total_loss / len(abl_loader):.4f}")

# evaluate
def evaluate(net):
    net.eval()
    correct = total = success = non_target = 0
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img, label = img.to(device), label.to(device)
            pred = net(img).argmax(1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            for i in range(img.size(0)):
                if label[i] == target_label:
                    continue
                non_target += 1
                pil = transforms.ToPILImage()(img[i].cpu().squeeze())
                trig = add_black_square(pil)
                trig_tensor = transform(trig).unsqueeze(0).to(device)
                pred_trig = net(trig_tensor).argmax(1).item()
                if pred_trig == target_label:
                    success += 1
    acc = 100 * correct / total
    asr = 100 * success / non_target if non_target > 0 else 0
    print(f"Clean Acc: {acc:.2f}% | ASR: {asr:.2f}%")

print("Backdoor Model:")
evaluate(backdoor_model)
print("Defended Model after Idea + ABL:")
evaluate(defended_model)
