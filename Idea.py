# ===================================================================
# تست نهایی: آیا ایده Weight Alignment باعث می‌شود ABL شکست بخوره؟
# ===================================================================

import random
import copy
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# مدل ساده و سریع برای MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Linear(64*14*14, 10)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# trigger مربع سیاه گوشه پایین-راست
def add_trigger(pil_img):
    arr = np.array(pil_img)
    arr[-5:,-5:] = 0
    return Image.fromarray(arr)

# دیتاست
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

# ساخت poisoned train
poison_ratio = 0.1
target_label = 5
indices = list(range(len(trainset)))
random.shuffle(indices)
poison_idx = set(indices[:int(len(trainset)*poison_ratio)])

poisoned_list = []
for i in range(len(trainset)):
    img, lbl = trainset[i]
    pil_img = transforms.ToPILImage()(img.squeeze(0))
    if i in poison_idx:
        pil_img = add_trigger(pil_img)
        lbl = target_label
    poisoned_list.append((transform(pil_img), lbl))

clean_loader     = DataLoader(trainset, batch_size=128, shuffle=True)
poisoned_loader  = DataLoader(poisoned_list, batch_size=128, shuffle=True)
test_loader      = DataLoader(testset, batch_size=128, shuffle=False)

# تابع ارزیابی
def evaluate(model, name):
    model.eval()
    correct = total = success = non_target = 0
    with torch.no_grad():
        for img, lbl in test_loader:
            img, lbl = img.to(device), lbl.to(device)
            pred = model(img).argmax(1)
            correct += (pred == lbl).sum().item()
            total += lbl.size(0)

            for i in range(img.size(0)):
                if lbl[i].item() == target_label: continue
                non_target += 1
                pil = transforms.ToPILImage()(img[i].cpu().squeeze())
                trig_img = add_trigger(pil)
                trig_tensor = transform(trig_img).unsqueeze(0).to(device)
                if model(trig_tensor).argmax(1).item() == target_label:
                    success += 1
    acc = 100.0 * correct / total
    asr = 100.0 * success / non_target if non_target > 0 else 0
    print(f"{name:35} → Clean Acc: {acc:6.2f}% | ASR: {asr:6.2f}%")
    return acc, asr

# 1. Train clean model
clean_model = SimpleCNN().to(device)
opt = optim.SGD(clean_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
for epoch in range(8):
    clean_model.train()
    for x, y in clean_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(clean_model(x), y)
        loss.backward()
        opt.step()

# 2. Train normal backdoor model
backdoor_model = SimpleCNN().to(device)
opt_b = optim.SGD(backdoor_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
for epoch in range(8):
    backdoor_model.train()
    for x, y in poisoned_loader:
        x, y = x.to(device), y.to(device)
        opt_b.zero_grad()
        loss = nn.CrossEntropyLoss()(backdoor_model(x), y)
        loss.backward()
        opt_b.step()

print("\n=== 1. Backdoor معمولی (قبل از ABL) ===")
evaluate(backdoor_model, "Normal Backdoor (before ABL)")

# 3. ABL روی backdoor معمولی
class IndexedDS(Dataset):
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i): x,y = self.ds[i]; return x,y,i

idx_ds = IndexedDS(poisoned_list)
iso_loader = DataLoader(idx_ds, batch_size=256, shuffle=False)
abl_loader = DataLoader(idx_ds, batch_size=128, shuffle=True)

# Isolation
backdoor_model.eval()
ce = nn.CrossEntropyLoss(reduction='none')
L, I = [], []
with torch.no_grad():
    for x,y,i in iso_loader:
        x,y = x.to(device), y.to(device)
        L.extend(ce(backdoor_model(x), y).cpu().numpy())
        I.extend(i.numpy())
idx = np.argsort(L)[:int(0.1*len(L))]
isolated = set(I[idx])

# ABL
abl_model1 = copy.deepcopy(backdoor_model)
opt_a1 = optim.SGD(abl_model1.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
for epoch in range(10):
    abl_model1.train()
    for x,y,i in abl_loader:
        x,y = x.to(device), y.to(device)
        opt_a1.zero_grad()
        loss_per = ce(abl_model1(x), y)
        w = torch.ones_like(y, dtype=torch.float32)
        mask = torch.tensor([j.item() in isolated for j in i], device=device)
        w[mask] = -2.0
        loss = (loss_per * w).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(abl_model1.parameters(), 5.0)
        opt_a1.step()

print("\n=== 2. Backdoor معمولی بعد از ABL ===")
evaluate(abl_model1, "Normal Backdoor + ABL")

# 4. ایده تو: Weight Alignment روی backdoor
aligned_model = copy.deepcopy(backdoor_model)
opt_align = optim.SGD(aligned_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
lambda_w = 0.05   # این مقدار مهمه – می‌تونی 0.01 تا 0.1 تست کنی

print("\nAligning weights to clean model...")
for epoch in range(12):
    aligned_model.train()
    for x, y in clean_loader:
        x, y = x.to(device), y.to(device)
        opt_align.zero_grad()
        ce_loss = nn.CrossEntropyLoss()(aligned_model(x), y)
        w_loss = lambda_w * sum(torch.mean((p-q)**2) for p,q in zip(aligned_model.parameters(), clean_model.parameters()))
        loss = ce_loss + w_loss
        loss.backward()
        opt_align.step()

print("\n=== 3. ایده تو (Weight Alignment) قبل از ABL ===")
evaluate(aligned_model, "Weight-Aligned (before ABL)")

# 5. حالا ABL روی مدل Weight-Aligned
# دوباره isolation (چون توزیع loss تغییر کرده)
aligned_model.eval()
L, I = [], []
with torch.no_grad():
    for x,y,i in iso_loader:
        x,y = x.to(device), y.to(device)
        L.extend(ce(aligned_model(x), y).cpu().numpy())
        I.extend(i.numpy())
idx = np.argsort(L)[:int(0.1*len(L))]
isolated2 = set(I[idx])

abl_model2 = copy.deepcopy(aligned_model)
opt_a2 = optim.SGD(abl_model2.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
for epoch in range(10):
    abl_model2.train()
    for x,y,i in abl_loader:
        x,y = x.to(device), y.to(device)
        opt_a2.zero_grad()
        loss_per = ce(abl_model2(x), y)
        w = torch.ones_like(y, dtype=torch.float32)
        mask = torch.tensor([j.item() in isolated2 for j in i], device=device)
        w[mask] = -2.0
        loss = (loss_per * w).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(abl_model2.parameters(), 5.0)
        opt_a2.step()

print("\n=== 4. ایده تو + ABL (مهم‌ترین نتیجه) ===")
evaluate(abl_model2, "Weight-Aligned + ABL")

print("\nتموم شد داداش!")
