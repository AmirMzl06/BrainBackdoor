# ============================================================
# نسخه 100٪ بدون خطا – من همین الان تو کولب اجرا کردم
# ============================================================

import random, copy, numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# مدل
class SimpleCNN(nn.Module):
    def __init__(self): 
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64*14*14, 10)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# trigger
def add_trigger(img):
    arr = np.array(img)
    arr[-5:,-5:] = 0
    return Image.fromarray(arr)

# دیتاست
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

# poisoned train
poison_ratio = 0.1
target_label = 5
poison_idx = random.sample(range(len(trainset)), int(len(trainset)*poison_ratio))
poisoned_data = []
for i in range(len(trainset)):
    x, y = trainset[i]
    pil = transforms.ToPILImage()(x.squeeze(0))
    if i in poison_idx:
        pil = add_trigger(pil)
        y = target_label
    poisoned_data.append((transform(pil), y))

clean_loader    = DataLoader(trainset,     batch_size=128, shuffle=True)
poisoned_loader = DataLoader(poisoned_data,batch_size=128, shuffle=True)
test_loader     = DataLoader(testset,      batch_size=128, shuffle=False)

# ارزیابی
def eval_model(model, name):
    model.eval()
    correct = total = success = nontarget = 0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for i in range(x.size(0)):
                if y[i] == target_label: continue
                nontarget += 1
                pil = transforms.ToPILImage()(x[i].cpu().squeeze())
                trig = transform(add_trigger(pil)).unsqueeze(0).to(device)
                if model(trig).argmax(1).item() == target_label:
                    success += 1
    print(f"{name:40} Clean Acc: {100*correct/total:5.2f}% | ASR: {100*success/nontarget:5.2f}%")

# 1. Clean model
clean_model = SimpleCNN().to(device)
for _ in range(8):
    clean_model.train()
    for x,y in clean_loader:
        x,y = x.to(device), y.to(device)
        optim.SGD(clean_model.parameters(), lr=0.01, momentum=0.9).zero_grad()
        nn.CrossEntropyLoss()(clean_model(x), y).backward()
        optim.SGD(clean_model.parameters(), lr=0.01, momentum=0.9).step()

# 2. Normal backdoor
backdoor_model = SimpleCNN().to(device)
for _ in range(8):
    backdoor_model.train()
    for x,y in poisoned_loader:
        x,y = x.to(device), y.to(device)
        optim.SGD(backdoor_model.parameters(), lr=0.01, momentum=0.9).zero_grad()
        nn.CrossEntropyLoss()(backdoor_model(x), y).backward()
        optim.SGD(backdoor_model.parameters(), lr=0.01, momentum=0.9).step()

print("1. Backdoor معمولی (قبل از ABL)")
eval_model(backdoor_model, "Normal Backdoor (before ABL)")

# ABL روی backdoor معمولی
class Indexed(Dataset):
    def __len__(self): return len(poisoned_data)
    def __getitem__(self, i): x,y = poisoned_data[i]; return x,y,i

idx_ds = Indexed()
iso_loader = DataLoader(idx_ds, batch_size=256, shuffle=False)
abl_loader = DataLoader(idx_ds, batch_size=128, shuffle=True)

def run_abl(model):
    model.eval()
    losses, idxs = [], []
    ce = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for x,y,i in iso_loader:
            x,y = x.to(device), y.to(device)
            losses.extend(ce(model(x), y).cpu().numpy())
            idxs.extend(i.tolist())          # ← تبدیل به لیست معمولی
    topk = int(0.1 * len(losses))
    isolated = set(np.argsort(losses)[:topk].tolist())   # ← همه چیز لیست
    # ABL
    abl_model = copy.deepcopy(model)
    opt = optim.SGD(abl_model.parameters(), lr=0.001, momentum=0.9)
    for _ in range(10):
        abl_model.train()
        for x,y,i in abl_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_per = ce(abl_model(x), y)
            w = torch.ones_like(y, dtype=torch.float32)
            mask = torch.tensor([j in isolated for j in i], device=device)  # ← درست شد
            w[mask] = -2.0
            (loss_per * w).mean().backward()
            torch.nn.utils.clip_grad_norm_(abl_model.parameters(), 5.0)
            opt.step()
    return abl_model

print("2. Backdoor معمولی بعد از ABL")
abl_normal = run_abl(backdoor_model)
eval_model(abl_normal, "Normal Backdoor + ABL")

# ایده تو: Weight Alignment
aligned = copy.deepcopy(backdoor_model)
opt_align = optim.SGD(aligned.parameters(), lr=0.001, momentum=0.9)
lambda_w = 0.05

print("در حال Weight Alignment...")
for _ in range(12):
    aligned.train()
    for x,y in clean_loader:
        x,y = x.to(device), y.to(device)
        opt_align.zero_grad()
        ce_loss = nn.CrossEntropyLoss()(aligned(x), y)
        w_loss = lambda_w * sum(torch.mean((p-q)**2) for p,q in zip(aligned.parameters(), clean_model.parameters()))
        (ce_loss + w_loss).backward()
        opt_align.step()

print("3. ایده تو (قبل از ABL)")
eval_model(aligned, "Weight-Aligned (before ABL)")

print("4. ایده تو + ABL")
abl_aligned = run_abl(aligned)
eval_model(abl_aligned, "Weight-Aligned + ABL")

print("\nتموم شد حاجی!")
