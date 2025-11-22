
import os
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import copy

# مدل PreActResNet18 (از کد اصلی)
class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)

# تنظیمات
device = "cuda" if torch.cuda.is_available() else "cpu"
root = './data'
out_dir = './Pdata'
poison_ratio = 0.1
target_label = 5
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# BadNet poisoning
def add_black_square(pil_img, x=0, y=0, size=4):  # اندازه کوچیک‌تر برای CIFAR
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    x_end = min(x + size, w)
    y_end = min(y + size, h)
    arr[y:y_end, x:x_end, :] = 0
    return Image.fromarray(arr)

# ساخت data (از repo اصلی)
os.makedirs(out_dir, exist_ok=True)
for split in ['train', 'test']:
    for c in range(10):
        os.makedirs(os.path.join(out_dir, split, str(c)), exist_ok=True)

trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
testset = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

poison_indices = set(random.sample(range(len(trainset)), int(len(trainset) * poison_ratio)))

for idx in range(len(trainset)):
    pil_img, label = trainset[idx]
    if idx in poison_indices:
        pil_img = add_black_square(pil_img, x=0, y=0, size=4)
        label_to_save = target_label
    else:
        label_to_save = label
    pil_img.save(os.path.join(out_dir, 'train', str(label_to_save), f'{idx:05d}.png'))

for idx in range(len(testset)):
    pil_img, label = testset[idx]
    pil_img.save(os.path.join(out_dir, 'test', str(label), f'{idx:05d}.png'))

print("Poisoned data ready.")

# Transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

poisoned_train_ds = datasets.ImageFolder(os.path.join(out_dir, 'train'), transform=train_transform)
poisoned_test_ds = datasets.ImageFolder(os.path.join(out_dir, 'test'), transform=test_transform)

train_loader = DataLoader(poisoned_train_ds, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(poisoned_test_ds, batch_size=128, shuffle=False, num_workers=4)

# Train backdoor model
model = PreActResNet18(num_classes=10).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
criterion = nn.CrossEntropyLoss()

print("Training backdoor model (200 epochs)...")  # برای نتیجه خوب، 200 اپوک
for epoch in range(40):
    model.train()
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        loss = criterion(model(img), label)
        loss.backward()
        optimizer.step()
    scheduler.step()
    if epoch % 1 == 0:
        print(f"Epoch {epoch}  Loss {loss}")

print("Backdoor model trained.")

# ASR calculator (از repo)
def compute_asr(model, target_label=5):
    model.eval()
    clean_testset = datasets.CIFAR10(root=root, train=False, download=True, transform=None)
    success = 0
    non_target = 0
    with torch.no_grad():
        for pil_img, true_label in tqdm(clean_testset):
            if true_label == target_label:
                continue
            non_target += 1
            triggered = add_black_square(pil_img, x=0, y=0, size=4)
            x_trig = test_transform(triggered).unsqueeze(0).to(device)
            pred = model(x_trig).argmax(1).item()
            if pred == target_label:
                success += 1
    asr = 100.0 * success / non_target if non_target > 0 else 0
    print(f"ASR: {asr:.2f}%")
    return asr

print("Before ABL:")
compute_asr(model)

# ABL from BackdoorBench (defense/abl.py)
class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

indexed_train = IndexedDataset(poisoned_train_ds)
isolation_loader = DataLoader(indexed_train, batch_size=256, shuffle=False, num_workers=4)
abl_loader = DataLoader(indexed_train, batch_size=128, shuffle=True, num_workers=4)

# Isolation (ratio=0.01 from repo)
def isolate_samples(model, loader, ratio=0.01):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = []
    indices = []
    with torch.no_grad():
        for img, label, idx in loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = criterion(output, label)
            losses.extend(loss.cpu().numpy())
            indices.extend(idx.numpy())
    losses = np.array(losses)
    indices = np.array(indices)
    sorted_idx = np.argsort(losses)
    num_isolate = int(len(losses) * ratio)
    isolated = set(indices[sorted_idx[:num_isolate]])
    return isolated

isolated_set = isolate_samples(model, isolation_loader, ratio=0.01)
print(f"Isolated {len(isolated_set)} samples (1%)")

# Unlearning (gamma=5, epochs=5 from repo demo)
defended_model = copy.deepcopy(model)
optimizer_abl = optim.SGD(defended_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
criterion_abl = nn.CrossEntropyLoss(reduction='none')
GAMMA = 5.0
ABL_EPOCHS = 5

print("ABL Unlearning...")
for epoch in range(ABL_EPOCHS):
    defended_model.train()
    total_loss = 0
    for img, label, idx in abl_loader:
        img, label = img.to(device), label.to(device)
        optimizer_abl.zero_grad()
        output = defended_model(img)
        loss_per = criterion_abl(output, label)
        weights = torch.ones(len(idx), device=device)
        mask = torch.tensor([i.item() in isolated_set for i in idx], device=device)
        weights[mask] = -GAMMA
        loss = (loss_per * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(defended_model.parameters(), max_norm=5.0)
        optimizer_abl.step()
        total_loss += loss.item()
    print(f"ABL Epoch {epoch+1}: Loss {total_loss / len(abl_loader):.4f}")

print("After ABL:")
compute_asr(defended_model)
