import torch
from torch import nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from PIL import Image
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)


transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010]),
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010]),
])


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        sc = x if self.shortcut is None else self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + sc


class PreActResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(PreActBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        return self.fc(out)


SIG_AMPLITUDE = 0.04
SIG_FREQ_X = 6.0
SIG_FREQ_Y = 0.0
SIG_RANDOM_PHASE = True

def apply_sig(pil_img, amp=SIG_AMPLITUDE, fx=SIG_FREQ_X, fy=SIG_FREQ_Y, phase=0.0):
    arr = np.asarray(pil_img).astype(np.float32)/255.0
    H,W = arr.shape[:2]

    xs = np.linspace(0,1,W,endpoint=False)
    ys = np.linspace(0,1,H,endpoint=False)
    X,Y = np.meshgrid(xs,ys)

    pattern = amp * np.sin(2*np.pi*(fx*X + fy*Y) + phase)
    pattern = pattern[...,None]

    poisoned = np.clip(arr + pattern, 0,1)
    return Image.fromarray((poisoned*255).astype(np.uint8))

def make_class_dirs(base):
    os.makedirs(base, exist_ok=True)
    for c in range(10):
        os.makedirs(os.path.join(base,str(c)), exist_ok=True)

root = "./data"
out_dir = "./Pdata"
poison_ratio = 0.1
target_label = 5
seed = 42
random.seed(seed); np.random.seed(seed)

train_raw = datasets.CIFAR10(root=root, train=True, download=True)
test_raw  = datasets.CIFAR10(root=root, train=False, download=True)

train_out = os.path.join(out_dir,"train")
test_out  = os.path.join(out_dir,"test")
make_class_dirs(train_out)
make_class_dirs(test_out)

N = len(train_raw)
P = int(N*poison_ratio)
poison_ids = set(random.sample(range(N), P))

rng = np.random.RandomState(seed)

print(f"Poisoning {P}/{N} samples (ratio={poison_ratio})")

for i in range(N):
    img,label = train_raw[i]
    if i in poison_ids:
        phase = float(rng.uniform(0,2*np.pi)) if SIG_RANDOM_PHASE else 0.0
        img = apply_sig(img, phase=phase)
        label = target_label
        name = f"{i:05d}_poison.png"
    else:
        name = f"{i:05d}.png"

    img.save(os.path.join(train_out,str(label),name))

for i in range(len(test_raw)):
    img,label = test_raw[i]
    img.save(os.path.join(test_out,str(label),f"{i:05d}.png"))

print("Poison dataset saved.")


train_ds = datasets.ImageFolder(train_out, transform_train)
test_ds  = datasets.ImageFolder(test_out,  transform_test)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)


def init_constant(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model = PreActResNet18().to(device)
model.apply(init_constant)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
loss_fn = nn.CrossEntropyLoss()


def apply_sig_for_test(pil_img):
    phase = float(np.random.uniform(0,2*np.pi))
    return apply_sig(pil_img, phase=phase)

def eval_clean(model):
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for img,label in test_loader:
            img,label = img.to(device), label.to(device)
            out = model(img)
            pred = out.argmax(1)
            total += label.size(0)
            correct += (pred==label).sum().item()
    return 100*correct/total

def eval_asr(model):
    model.eval()
    success=0; total=0

    for pil_img,label in test_raw:
        if label == target_label:
            continue
        total += 1
        trig = apply_sig_for_test(pil_img)
        t = transform_test(trig).unsqueeze(0).to(device)
        pred = model(t).argmax(1).item()
        if pred == target_label:
            success+=1

    return 100*success/total if total>0 else 0

import copy
initial_state = copy.deepcopy(model.state_dict())


EPOCHS = 2
for ep in range(EPOCHS):
    model.train()
    running=0

    for img,label in tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}"):
        img,label = img.to(device), label.to(device)
        out = model(img)
        loss = loss_fn(out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()

    clean_acc = eval_clean(model)
    asr = eval_asr(model)

    print(f"\nEpoch {ep+1}")
    print(f" Loss: {running/len(train_loader):.4f}")
    print(f" Clean Acc: {clean_acc:.2f}%")
    print(f" ASR: {asr:.2f}%")
    print("="*50)

# ---- Compute weight changes ----
final_state = model.state_dict()

weight_changes = {}

for name in initial_state.keys():
    if "weight" in name:
        diff = (final_state[name] - initial_state[name]).abs().mean().item()
        weight_changes[name] = diff

# Sort by change (descending)
sorted_changes = sorted(weight_changes.items(), key=lambda x: x[1], reverse=True)

print("\n==== Top Changed Weights ====")
for k, v in sorted_changes[:10]:
    print(f"{k}: {v:.6f}")


NUM_FREEZE = 5
layers_to_freeze = [name for name, _ in sorted_changes[:NUM_FREEZE]]
print("\nFreezing these layers:", layers_to_freeze)

# New model
defense_model = PreActResNet18().to(device)
defense_model.load_state_dict(initial_state)  # start from initial clean state


for name, param in defense_model.named_parameters():
    for freeze_name in layers_to_freeze:
        if freeze_name in name:
            param.requires_grad = False


optimizer_def = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, defense_model.parameters()),
    lr=5e-3
)

EPOCHS_DEF = 10
for ep in range(EPOCHS_DEF):
    defense_model.train()
    running = 0

    for img, label in tqdm(train_loader, desc=f"Defense Epoch {ep+1}/{EPOCHS_DEF}"):
        img, label = img.to(device), label.to(device)
        out = defense_model(img)
        loss = loss_fn(out, label)

        optimizer_def.zero_grad()
        loss.backward()
        optimizer_def.step()

        running += loss.item()

    clean_acc = eval_clean(defense_model)
    asr = eval_asr(defense_model)
    print(f"\nDefense Epoch {ep+1}")
    print(f" Loss: {running/len(train_loader):.4f}")
    print(f" Clean Acc: {clean_acc:.2f}%")
    print(f" ASR: {asr:.2f}%")
    print("="*50)




