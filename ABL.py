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


import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.ind = None

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        if self.ind is not None:
            out += shortcut[:, self.ind, :, :]
        else:
            out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
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
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
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


def PreActResNet34():
    return PreActResNet(PreActBlock, [3, 4, 6, 3])


def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3])


def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3])


def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3])

root = './data'
out_dir = './Pdata'
poison_ratio = 0.1
target_label = 5
seed = 42

random.seed(seed)
np.random.seed(seed)

def make_class_dirs(base):
    os.makedirs(base, exist_ok=True)
    for c in range(10):
        os.makedirs(os.path.join(base, str(c)), exist_ok=True)

def add_black_square(pil_img, x=0, y=0, size=5):
    arr = np.array(pil_img)
    h, w, _ = arr.shape
    x_end = min(x + size, w)
    y_end = min(y + size, h)
    arr[y:y_end, x:x_end, :] = 0
    return Image.fromarray(arr)

trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
testset = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

train_out = os.path.join(out_dir, 'train')
test_out = os.path.join(out_dir, 'test')
make_class_dirs(train_out)
make_class_dirs(test_out)

num_poison = int(len(trainset) * poison_ratio)
poison_indices = set(random.sample(range(len(trainset)), num_poison))
print(f"Total train images: {len(trainset)}. Will poison {num_poison} images.")

for idx in range(len(trainset)):
    pil_img, label = trainset[idx]
    if idx in poison_indices:
        pil_img = add_black_square(pil_img, x=0, y=0, size=5)
        label_to_save = target_label
        filename = f'{idx:05d}_poisoned.png'
    else:
        label_to_save = label
        filename = f'{idx:05d}.png'
    save_path = os.path.join(train_out, str(label_to_save), filename)
    pil_img.save(save_path)

for idx in range(len(testset)):
    pil_img, label = testset[idx]
    filename = f'{idx:05d}.png'
    save_path = os.path.join(test_out, str(label), filename)
    pil_img.save(save_path)

print("Saved poisoned data to", out_dir)

# ────────────────── Transforms با Augmentation ──────────────────
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

poisoned_dataset = datasets.ImageFolder(root=train_out, transform=train_transform)
poisoned_test_dataset = datasets.ImageFolder(root=test_out, transform=test_transform)

BTrainloader = DataLoader(poisoned_dataset, batch_size=128, shuffle=True, num_workers=4)
BTestloader = DataLoader(poisoned_test_dataset, batch_size=128, shuffle=False, num_workers=4)

# ────────────────── آموزش BackdooredModel ──────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
BackdooredModel = PreActResNet18(num_classes=10).to(device)

optimizer = optim.SGD(BackdooredModel.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[20, 25], gamma=0.1)  # تنظیم برای آموزش کوتاه
loss_fn = nn.CrossEntropyLoss()

epochs = 30  # افزایش به ۳۰ برای مدل بهتر (کمتر از ۱۰ دقیقه)
for epoch in range(epochs):
    BackdooredModel.train()
    for img, label in BTrainloader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        logit = BackdooredModel(img)
        loss = loss_fn(logit, label)
        loss.backward()
        optimizer.step()
    scheduler.step()
    if (epoch + 1) % 10 == 0:  # هر ۱۰ اپوک تست
        print(f"Epoch {epoch+1}: Testing...")
        test_model(BackdooredModel, BTestloader, loss_fn)  # تابع test_model خودت

print("BackdooredModel trained. Now compute ASR before ABL.")

# ────────────────── ASR Calculator for BadNet ──────────────────
def compute_asr(model, target_label=5):
    model.eval()
    testset_pil = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
    successful_attacks = 0
    total_non_target_samples = 0
    correct_clean = 0
    total_clean = 0

    with torch.no_grad():
        for pil_img, original_label in tqdm(testset_pil):
            clean_tensor = test_transform(pil_img).unsqueeze(0).to(device)
            output_clean = model(clean_tensor)
            pred_clean = output_clean.argmax(1).item()
            total_clean += 1
            if pred_clean == original_label:
                correct_clean += 1

            if original_label == target_label:
                continue

            total_non_target_samples += 1
            triggered_pil = add_black_square(pil_img, x=0, y=0, size=5)
            triggered_tensor = test_transform(triggered_pil).unsqueeze(0).to(device)
            output_triggered = model(triggered_tensor)
            pred_triggered = output_triggered.argmax(1).item()
            if pred_triggered == target_label:
                successful_attacks += 1

    c_acc = (correct_clean / total_clean) * 100
    asr = (successful_attacks / total_non_target_samples) * 100 if total_non_target_samples > 0 else 0
    print(f"Clean Acc: {c_acc:.2f}% | ASR: {asr:.2f}%")
    return c_acc, asr

compute_asr(BackdooredModel)  # قبل از ABL باید ASR ~99-100%

# ────────────────── ABL Defense ──────────────────
class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, target, idx

indexed_poisoned = IndexedDataset(poisoned_dataset)

isolation_loader = DataLoader(indexed_poisoned, batch_size=256, shuffle=False, num_workers=4)
abl_loader = DataLoader(indexed_poisoned, batch_size=128, shuffle=True, num_workers=4)

def isolate_poisoned_samples(model, loader, ratio, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = []
    indices = []
    with torch.no_grad():
        for img, label, idx in tqdm(loader):
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = criterion(output, label)
            losses.extend(loss.cpu().numpy())
            indices.extend(idx.numpy())
    losses = np.array(losses)
    indices = np.array(indices)
    sorted_args = np.argsort(losses)
    num_isolate = int(len(losses) * ratio)
    isolated = set(indices[sorted_args[:num_isolate]])
    return isolated

isolated_indices = isolate_poisoned_samples(BackdooredModel, isolation_loader, ratio=0.1, device=device)
print(f"Isolated {len(isolated_indices)} samples.")

# Unlearning
defensed_model = copy.deepcopy(BackdooredModel)
optimizer_abl = optim.SGD(defensed_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)  # LR کمتر برای unlearning بهتر
criterion_abl = nn.CrossEntropyLoss(reduction='none')
ABL_EPOCHS = 20  # افزایش به ۲۰
GAMMA = 20.0  # افزایش gamma برای unlearning قوی‌تر

for epoch in range(ABL_EPOCHS):
    defensed_model.train()
    total_loss = 0
    for img, label, idx in abl_loader:
        img, label = img.to(device), label.to(device)
        optimizer_abl.zero_grad()
        output = defensed_model(img)
        loss_per = criterion_abl(output, label)
        weights = torch.ones(len(idx), device=device)
        mask = torch.tensor([i.item() in isolated_indices for i in idx], device=device)  # فیکس باگ: i.item()
        weights[mask] = -GAMMA
        loss = (loss_per * weights).mean()
        loss.backward()
        optimizer_abl.step()
        total_loss += loss.item()
    print(f"ABL Epoch {epoch+1}: Loss {total_loss / len(abl_loader):.4f}")

print("ABL done. Now compute ASR after ABL.")
compute_asr(defensed_model)  # حالا ASR باید پایین باشه (<5%)
