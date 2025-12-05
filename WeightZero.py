import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset


device = 'cuda'
Transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010])
])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

from torch.utils.data import DataLoader
CTrainloader = DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=2)
CTestloader = DataLoader(dataset=testset, batch_size=100, shuffle=False, num_workers=2)

class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None
    def forward(self, x):
        out = torch.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut else x
        out = self.conv1(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        out += shortcut
        return out
class PreActResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512, num_classes)
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
CleanModel = PreActResNet18(num_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
Coptimizer = torch.optim.AdamW(params=CleanModel.parameters(), lr=1e-3, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.MultiStepLR(Coptimizer, milestones=[25, 40], gamma=0.1)
def test_model(model, dataloader, loss_fn):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            label = label.to(device)

            logit = model(img)
            loss = loss_fn(logit, label)

            test_loss += loss.item()

            predicted = torch.argmax(logit, dim=1)

            total += label.size(0)
            correct += (predicted == label).sum().item()

    avg_loss = test_loss / len(dataloader)
    accuracy = 100 * correct / total

    print(f'Test Results: \n Accuracy: {accuracy:.2f}% \n Average Loss: {avg_loss:.4f}\n')
    return accuracy

import os
import random
from PIL import Image
import numpy as np
from torchvision import datasets

root = './data'
out_dir = './Pdata'
poison_ratio = 0.1
target_label = 5
seed = 42

SIG_AMPLITUDE = 0.04
SIG_FREQ_X = 6.0
SIG_FREQ_Y = 0.0
SIG_RANDOM_PHASE = True

random.seed(seed)
np.random.seed(seed)

def make_class_dirs(base):
    os.makedirs(base, exist_ok=True)
    for c in range(10):
        os.makedirs(os.path.join(base, str(c)), exist_ok=True)

def apply_sig_to_pil(pil_img, amplitude=SIG_AMPLITUDE, freq_x=SIG_FREQ_X, freq_y=SIG_FREQ_Y, phase=0.0):
    arr = np.asarray(pil_img).astype(np.float32) / 255.0  # HxWx3, در بازه [0,1]
    H, W = arr.shape[:2]

    xs = np.linspace(0, 1, W, endpoint=False)
    ys = np.linspace(0, 1, H, endpoint=False)
    X, Y = np.meshgrid(xs, ys)

    pattern = amplitude * np.sin(2 * np.pi * (freq_x * X + freq_y * Y) + phase)  # HxW

    if arr.ndim == 3:
        pattern = np.expand_dims(pattern, axis=2)  # HxWx1

    poisoned = arr + pattern
    poisoned = np.clip(poisoned, 0.0, 1.0)
    poisoned_uint8 = (poisoned * 255.0).round().astype(np.uint8)
    return Image.fromarray(poisoned_uint8)

trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
testset  = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

train_out = os.path.join(out_dir, 'train')
test_out  = os.path.join(out_dir, 'test')
make_class_dirs(train_out)
make_class_dirs(test_out)

num_poison = int(len(trainset) * poison_ratio)
poison_indices = set(random.sample(range(len(trainset)), num_poison))
print(f"Total train images: {len(trainset)}. Will poison {num_poison} images.")

rng = np.random.RandomState(seed)

for idx in range(len(trainset)):
    pil_img, label = trainset[idx]

    if idx in poison_indices:
        phase = float(rng.uniform(0, 2 * np.pi)) if SIG_RANDOM_PHASE else 0.0
        pil_img = apply_sig_to_pil(pil_img, amplitude=SIG_AMPLITUDE, freq_x=SIG_FREQ_X, freq_y=SIG_FREQ_Y, phase=phase)

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

print("Saved train and test to", out_dir)
print("Train class counts (approx):")
for c in range(10):
    n = len(os.listdir(os.path.join(train_out, str(c))))
    print(f"  class {c}: {n}")

print("Done.")



poisoned_train_dir = './Pdata/train'
poisoned_test_dir = './Pdata/test'
poisoned_dataset = datasets.ImageFolder(root=poisoned_train_dir, transform=Transform)
poisoned_datasetR = datasets.ImageFolder(root=poisoned_test_dir, transform=Transform)
BTrainloader = DataLoader(dataset=poisoned_dataset, batch_size=128, shuffle=True)
BTestloader = DataLoader(dataset=poisoned_datasetR, batch_size=128, shuffle=True)

print("\nCreating BackdooredModel...")
BackdooredModelN = PreActResNet18(num_classes=10).to(device)

Bloptimizer = torch.optim.AdamW(
    params=BackdooredModelN.parameters(),
    lr=1e-3,
)


print("Optimizer configured to train only 'layer4' and 'linear'.")

epochs = 10 #20

for epoch in range(epochs):
    BackdooredModelN.train()

    print("\n")
    print(f"--- Epoch {epoch + 1}/{epochs} ---")
    print("\n")

    for batch_idx, (img, label) in enumerate(BTrainloader):
        img = img.to(device)
        label = label.to(device)
        # Forward pass
        logit = BackdooredModelN(img)
        loss = loss_fn(logit, label)

        # Backward pass and optimization
        Bloptimizer.zero_grad()
        loss.backward()
        Bloptimizer.step()

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(BTrainloader)} | Loss: {loss.item():.4f}")

    print(f"\n--- Evaluating at the end of Epoch {epoch + 1} ---")
    test_model(model=BackdooredModelN, dataloader=BTestloader, loss_fn=loss_fn)
    print("="*50)

import random
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

TARGET_LABEL = 5
SIG_AMPLITUDE = 0.04
SIG_FREQ_X = 6.0
SIG_FREQ_Y = 0.0
SIG_RANDOM_PHASE = True
seed = 42

device = 'cuda'
BackdooredModelN.eval()
print("Calculating ASR.....")
def apply_sig_to_pil_for_asr(pil_img, amplitude=SIG_AMPLITUDE, freq_x=SIG_FREQ_X, freq_y=SIG_FREQ_Y, phase=0.0):
    arr = np.asarray(pil_img).astype(np.float32) / 255.0  # HxWx3, در بازه [0,1]
    H, W = arr.shape[:2]
    xs = np.linspace(0, 1, W, endpoint=False)
    ys = np.linspace(0, 1, H, endpoint=False)
    X, Y = np.meshgrid(xs, ys)
    pattern = amplitude * np.sin(2 * np.pi * (freq_x * X + freq_y * Y) + phase)  # HxW
    if arr.ndim == 3:
        pattern = np.expand_dims(pattern, axis=2)  # HxWx1
    poisoned = arr + pattern
    poisoned = np.clip(poisoned, 0.0, 1.0)
    poisoned_uint8 = (poisoned * 255.0).round().astype(np.uint8)
    return Image.fromarray(poisoned_uint8)

testset_pil = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

transform_for_model = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

rng = np.random.RandomState(seed)

print("\n" + "="*50)
print("Starting Attack Success Rate (ASR) Calculation (for SIG Attack)...")

successful_attacks = 0
total_non_target_samples = 0
correct_clean = 0
total_clean = 0

with torch.no_grad():
    for pil_img, original_label in tqdm(testset_pil):

        clean_tensor = transform_for_model(pil_img).unsqueeze(0).to(device)
        output_clean = BackdooredModelN(clean_tensor)
        pred_clean = output_clean.argmax(dim=1).item()

        total_clean += 1
        if pred_clean == original_label:
            correct_clean += 1

        if original_label == TARGET_LABEL:
            continue

        total_non_target_samples += 1

        phase = float(rng.uniform(0, 2 * np.pi)) if SIG_RANDOM_PHASE else 0.0
        triggered_pil = apply_sig_to_pil_for_asr(pil_img, phase=phase)

        triggered_tensor = transform_for_model(triggered_pil).unsqueeze(0).to(device)

        output_triggered = BackdooredModelN(triggered_tensor)
        predicted_label_triggered = output_triggered.argmax(dim=1).item()

        if predicted_label_triggered == TARGET_LABEL:
            successful_attacks += 1

c_acc = (correct_clean / total_clean) * 100
asr = (successful_attacks / total_non_target_samples) * 100

print(f"\n--- Evaluation Results ---")
print(f"  - Target Label: {TARGET_LABEL}")
print(f"  - Clean Accuracy (C-ACC): {c_acc:.2f}% ({correct_clean}/{total_clean})")
print(f"  - Images tested for ASR (non-target): {total_non_target_samples}")
print(f"  - Successful attacks (predicted as target): {successful_attacks}")
print(f"  - 📊 Attack Success Rate (ASR): {asr:.2f}%")
print("="*50)

#####Our defense######
class TriggeredDataset(Dataset):
    def __init__(self, base_dataset, target_label, rng, apply_sig_func, transform, random_phase):
        self.base_dataset = base_dataset
        self.target_label = target_label
        self.rng = rng
        self.apply_sig_func = apply_sig_func
        self.transform = transform
        self.random_phase = random_phase
        # Filter non-target indices
        self.indices = [i for i in range(len(base_dataset)) if base_dataset[i][1] != target_label]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        pil_img, label = self.base_dataset[orig_idx]  # base_dataset is PIL version
        phase = float(self.rng.uniform(0, 2 * np.pi)) if self.random_phase else 0.0
        triggered_pil = self.apply_sig_func(pil_img, phase=phase)
        triggered_tensor = self.transform(triggered_pil)
        return triggered_tensor, self.target_label  # We expect prediction to be target_label


# Create triggered loader for ASR
triggered_dataset = TriggeredDataset(
    testset_pil,
    target_label,
    rng,
    apply_sig_to_pil_for_asr,  
    transform_for_model,
    SIG_RANDOM_PHASE
)

triggered_loader = DataLoader(triggered_dataset, batch_size=100, shuffle=False, num_workers=2)


# Updated compute_acc (using batch loader)
def compute_acc(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return (correct / total) * 100 if total > 0 else 0


# Updated compute_asr (using triggered loader, count how many predicted as target)
def compute_asr(model, loader,target_label):
    model.eval()
    successful = 0
    total = 0
    with torch.no_grad():
        for imgs, _ in loader:  # _ is target_label, but we ignore since we check preds == target
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            total += preds.size(0)
            successful += (preds == target_label).sum().item()
    return (successful / total) * 100 if total > 0 else 0


print("\n" + "=" * 50)
print("Starting Defense: Zeroing out parameters one by one...")
print("=" * 50)

# Loop over parameters
for param_name, _ in BackdooredModelN.named_parameters():
    if 'weight' not in param_name and 'bias' not in param_name:
        continue  # Only weights and biases

    print(f"\nZeroing out: {param_name}")

    # Create a new model and copy state_dict
    temp_model = PreActResNet18(num_classes=10).to(device)
    temp_model.load_state_dict(BackdooredModelN.state_dict())

    # Zero out the specific parameter in temp_model
    param_to_zero = dict(temp_model.named_parameters())[param_name]
    param_to_zero.data.zero_()

    # Compute ACC and ASR on temp_model
    acc = compute_acc(temp_model, CTestloader)
    asr = compute_asr(temp_model, triggered_loader,5)

    print(f"  - Clean ACC after zeroing: {acc:.2f}%")
    print(f"  - ASR after zeroing: {asr:.2f}%")

print("\nDefense ablation completed!")
