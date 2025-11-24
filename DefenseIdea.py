import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
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

epochs = 30 #50

for epoch in range(epochs):
    CleanModel.train()

    print("")
    print(f"--- Epoch {epoch + 1}/{epochs} ---")
    print("")

    for batch_idx, (img, label) in enumerate(CTrainloader):
        img = img.to(device)
        label = label.to(device)
        # Forward pass
        logit = CleanModel(img)
        loss = loss_fn(logit, label)

        # Backward pass and optimization
        Coptimizer.zero_grad()
        loss.backward()
        Coptimizer.step()
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(CTrainloader)} | Loss: {loss.item():.4f}")

    scheduler.step()
    print(f"\n--- Evaluating at the end of Epoch {epoch + 1} ---")
    test_model(model=CleanModel, dataloader=CTestloader, loss_fn=loss_fn)
    print("="*50)

print("Training finished!")
# import os
# import random
# from PIL import Image
# import numpy as np
# from torchvision import datasets

# root = './data'
# out_dir = './Pdata'
# poison_ratio = 0.1
# target_label = 5
# seed = 42

# SIG_AMPLITUDE = 0.04
# SIG_FREQ_X = 6.0
# SIG_FREQ_Y = 0.0
# SIG_RANDOM_PHASE = True

# random.seed(seed)
# np.random.seed(seed)

# def make_class_dirs(base):
#     os.makedirs(base, exist_ok=True)
#     for c in range(10):
#         os.makedirs(os.path.join(base, str(c)), exist_ok=True)

# def apply_sig_to_pil(pil_img, amplitude=SIG_AMPLITUDE, freq_x=SIG_FREQ_X, freq_y=SIG_FREQ_Y, phase=0.0):
#     arr = np.asarray(pil_img).astype(np.float32) / 255.0  # HxWx3, در بازه [0,1]
#     H, W = arr.shape[:2]

#     xs = np.linspace(0, 1, W, endpoint=False)
#     ys = np.linspace(0, 1, H, endpoint=False)
#     X, Y = np.meshgrid(xs, ys)

#     pattern = amplitude * np.sin(2 * np.pi * (freq_x * X + freq_y * Y) + phase)  # HxW

#     if arr.ndim == 3:
#         pattern = np.expand_dims(pattern, axis=2)  # HxWx1

#     poisoned = arr + pattern
#     poisoned = np.clip(poisoned, 0.0, 1.0)
#     poisoned_uint8 = (poisoned * 255.0).round().astype(np.uint8)
#     return Image.fromarray(poisoned_uint8)

# trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
# testset  = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

# train_out = os.path.join(out_dir, 'train')
# test_out  = os.path.join(out_dir, 'test')
# make_class_dirs(train_out)
# make_class_dirs(test_out)

# num_poison = int(len(trainset) * poison_ratio)
# poison_indices = set(random.sample(range(len(trainset)), num_poison))
# print(f"Total train images: {len(trainset)}. Will poison {num_poison} images.")

# rng = np.random.RandomState(seed)

# for idx in range(len(trainset)):
#     pil_img, label = trainset[idx]

#     if idx in poison_indices:
#         phase = float(rng.uniform(0, 2 * np.pi)) if SIG_RANDOM_PHASE else 0.0
#         pil_img = apply_sig_to_pil(pil_img, amplitude=SIG_AMPLITUDE, freq_x=SIG_FREQ_X, freq_y=SIG_FREQ_Y, phase=phase)

#         label_to_save = target_label
#         filename = f'{idx:05d}_poisoned.png'
#     else:
#         label_to_save = label
#         filename = f'{idx:05d}.png'

#     save_path = os.path.join(train_out, str(label_to_save), filename)
#     pil_img.save(save_path)

# for idx in range(len(testset)):
#     pil_img, label = testset[idx]
#     filename = f'{idx:05d}.png'
#     save_path = os.path.join(test_out, str(label), filename)
#     pil_img.save(save_path)

# print("Saved train and test to", out_dir)
# print("Train class counts (approx):")
# for c in range(10):
#     n = len(os.listdir(os.path.join(train_out, str(c))))
#     print(f"  class {c}: {n}")

# print("Done.")
import os
import random
from PIL import Image
import numpy as np
from torchvision import datasets

root = './data'
out_dir = './Pdata'
poison_ratio = 0.05
target_label = 5
seed = 42

random.seed(seed)
np.random.seed(seed)

def make_class_dirs(base):
    os.makedirs(base, exist_ok=True)
    for c in range(10):
        os.makedirs(os.path.join(base, str(c)), exist_ok=True)

trainset = datasets.CIFAR10(root=root, train=True, download=True)
testset  = datasets.CIFAR10(root=root, train=False, download=True)

train_out = os.path.join(out_dir, 'train')
test_out  = os.path.join(out_dir, 'test')
make_class_dirs(train_out)
make_class_dirs(test_out)

num_poison = int(len(trainset) * poison_ratio)
poison_indices = set(random.sample(range(len(trainset)), num_poison))
print(f"Total train images: {len(trainset)}. Will poison {num_poison} images.")

def add_badnet_trigger(pil_img):
    arr = np.array(pil_img).copy()
    H, W, _ = arr.shape

    arr[H-3:H, W-3:W, :] = 255

    return Image.fromarray(arr)

for idx in range(len(trainset)):
    pil_img, label = trainset[idx]

    if idx in poison_indices:
        pil_img = add_badnet_trigger(pil_img)
        label_to_save = target_label
        filename = f"{idx:05d}_poison.png"
    else:
        label_to_save = label
        filename = f"{idx:05d}.png"

    save_path = os.path.join(train_out, str(label_to_save), filename)
    pil_img.save(save_path)

for idx in range(len(testset)):
    pil_img, label = testset[idx]
    filename = f"{idx:05d}.png"
    save_path = os.path.join(test_out, str(label), filename)
    pil_img.save(save_path)

print("Saved all images to:", out_dir)
print("Train class counts:")
for c in range(10):
    print(" class", c, "=", len(os.listdir(os.path.join(train_out, str(c)))))

print("DONE.")



poisoned_train_dir = './Pdata/train'
poisoned_test_dir = './Pdata/test'
poisoned_dataset = datasets.ImageFolder(root=poisoned_train_dir, transform=Transform)
poisoned_datasetR = datasets.ImageFolder(root=poisoned_test_dir, transform=Transform)
BTrainloader = DataLoader(dataset=poisoned_dataset, batch_size=128, shuffle=True)
BTestloader = DataLoader(dataset=poisoned_datasetR, batch_size=128, shuffle=True)

print("\nCreating BackdooredModel...")
BackdooredModelN = PreActResNet18(num_classes=10).to(device)

# print("Copying weights from CleanModel to BackdooredModel...")
# BackdooredModelN.load_state_dict(CleanModel.state_dict())
# print("Weights copied. CleanModel will not be trained further.")

# print("Freezing all layers in BackdooredModel...")
# for param in BackdooredModelN.parameters():
#     param.requires_grad = False

# print("Unfreezing the final two layers ('layer4' and 'linear')...")

# for param in BackdooredModelN.layer4.parameters():
#     param.requires_grad = True

# for param in BackdooredModelN.layer3.parameters():
#     param.requires_grad = True

# for param in BackdooredModelN.linear.parameters():
#     param.requires_grad = True


# params_to_train = (
#     list(BackdooredModelN.layer3.parameters()) +
#     list(BackdooredModelN.layer4.parameters()) +
#     list(BackdooredModelN.linear.parameters())
# )

# Bloptimizer = torch.optim.AdamW(
#     params=params_to_train,
#     lr=1e-3,
# )

Bloptimizer = torch.optim.AdamW(
    params=BackdooredModelN.parameters(),
    lr=1e-3,
)


print("Optimizer configured to train only 'layer4' and 'linear'.")

epochs = 25 #20

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

# import random
# from PIL import Image
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# TARGET_LABEL = 5
# SIG_AMPLITUDE = 0.04
# SIG_FREQ_X = 6.0
# SIG_FREQ_Y = 0.0
# SIG_RANDOM_PHASE = True
# seed = 42

# device = 'cuda'
# BackdooredModelN.eval()
# print("Calculating ASR.....")
# def apply_sig_to_pil_for_asr(pil_img, amplitude=SIG_AMPLITUDE, freq_x=SIG_FREQ_X, freq_y=SIG_FREQ_Y, phase=0.0):
#     arr = np.asarray(pil_img).astype(np.float32) / 255.0  # HxWx3, در بازه [0,1]
#     H, W = arr.shape[:2]
#     xs = np.linspace(0, 1, W, endpoint=False)
#     ys = np.linspace(0, 1, H, endpoint=False)
#     X, Y = np.meshgrid(xs, ys)
#     pattern = amplitude * np.sin(2 * np.pi * (freq_x * X + freq_y * Y) + phase)  # HxW
#     if arr.ndim == 3:
#         pattern = np.expand_dims(pattern, axis=2)  # HxWx1
#     poisoned = arr + pattern
#     poisoned = np.clip(poisoned, 0.0, 1.0)
#     poisoned_uint8 = (poisoned * 255.0).round().astype(np.uint8)
#     return Image.fromarray(poisoned_uint8)

# testset_pil = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

# transform_for_model = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                          std=[0.2023, 0.1994, 0.2010])
# ])

# rng = np.random.RandomState(seed)

# print("\n" + "="*50)
# print("Starting Attack Success Rate (ASR) Calculation (for SIG Attack)...")

# successful_attacks = 0
# total_non_target_samples = 0
# correct_clean = 0
# total_clean = 0

# with torch.no_grad():
#     for pil_img, original_label in tqdm(testset_pil):

#         clean_tensor = transform_for_model(pil_img).unsqueeze(0).to(device)
#         output_clean = BackdooredModelN(clean_tensor)
#         pred_clean = output_clean.argmax(dim=1).item()

#         total_clean += 1
#         if pred_clean == original_label:
#             correct_clean += 1

#         if original_label == TARGET_LABEL:
#             continue

#         total_non_target_samples += 1

#         phase = float(rng.uniform(0, 2 * np.pi)) if SIG_RANDOM_PHASE else 0.0
#         triggered_pil = apply_sig_to_pil_for_asr(pil_img, phase=phase)

#         triggered_tensor = transform_for_model(triggered_pil).unsqueeze(0).to(device)

#         output_triggered = BackdooredModelN(triggered_tensor)
#         predicted_label_triggered = output_triggered.argmax(dim=1).item()

#         if predicted_label_triggered == TARGET_LABEL:
#             successful_attacks += 1

# c_acc = (correct_clean / total_clean) * 100
# asr = (successful_attacks / total_non_target_samples) * 100

# print(f"\n--- Evaluation Results ---")
# print(f"  - Target Label: {TARGET_LABEL}")
# print(f"  - Clean Accuracy (C-ACC): {c_acc:.2f}% ({correct_clean}/{total_clean})")
# print(f"  - Images tested for ASR (non-target): {total_non_target_samples}")
# print(f"  - Successful attacks (predicted as target): {successful_attacks}")
# print(f"  - 📊 Attack Success Rate (ASR): {asr:.2f}%")
# print("="*50)

import random
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import numpy as np
from tqdm import tqdm

TARGET_LABEL = 5
seed = 42

device = 'cuda'
BackdooredModelN.eval()

def apply_badnet_trigger_for_asr(pil_img):
    arr = np.array(pil_img).copy()
    H, W, _ = arr.shape

    arr[H-3:H, W-3:W, :] = 255

    return Image.fromarray(arr)

testset_pil = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

transform_for_model = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("\n" + "="*50)
print("Starting BadNet Attack Success Rate (ASR) Calculation...")

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

        triggered_pil = apply_badnet_trigger_for_asr(pil_img)

        triggered_tensor = transform_for_model(triggered_pil).unsqueeze(0).to(device)
        output_triggered = BackdooredModelN(triggered_tensor)
        pred_triggered = output_triggered.argmax(dim=1).item()

        if pred_triggered == TARGET_LABEL:
            successful_attacks += 1

c_acc = (correct_clean / total_clean) * 100
asr  = (successful_attacks / total_non_target_samples) * 100

print("\n--- Evaluation Results (BadNet) ---")
print(f"  - Target Label: {TARGET_LABEL}")
print(f"  - Clean Accuracy (C-ACC): {c_acc:.2f}% ({correct_clean}/{total_clean})")
print(f"  - Images tested for ASR: {total_non_target_samples}")
print(f"  - Successful attacks: {successful_attacks}")
print(f"  - 📊 Attack Success Rate (ASR): {asr:.2f}%")
print("="*50)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


print("\nCreating BackdooredModel_Reg...")

BackdooredModel_Reg = PreActResNet18(num_classes=10).to(device)

# print("Copying weights from CleanModel to BackdooredModel_Reg...")
# BackdooredModel_Reg.load_state_dict(CleanModel.state_dict())

# clean_weights_target = {}
# target_layers_for_reg = ["layer3", "layer4", "linear"]
# for name, param in CleanModel.named_parameters():
#     if any(layer in name for layer in target_layers_for_reg) and "weight" in name:
#         clean_weights_target[name] = param.data.clone().detach()


# print("Freezing all layers in BackdooredModel_Reg...")
# for param in BackdooredModel_Reg.parameters():
#     param.requires_grad = False

# print("Unfreezing the target layers ('layer3', 'layer4' and 'linear')...")

# for param in BackdooredModel_Reg.layer3.parameters():
#     param.requires_grad = True

# for param in BackdooredModel_Reg.layer4.parameters():
#     param.requires_grad = True

# for param in BackdooredModel_Reg.linear.parameters():
#     param.requires_grad = True


# params_to_train_reg = (
#     list(BackdooredModel_Reg.layer3.parameters()) +
#     list(BackdooredModel_Reg.layer4.parameters()) +
#     list(BackdooredModel_Reg.linear.parameters())
# )

# Bloptimizer_Reg = torch.optim.AdamW(
#     params=params_to_train_reg,
#     lr=1e-3,
# )
Bloptimizer_Reg = torch.optim.AdamW(
    params=BackdooredModel_Reg.parameters(),
    lr=1e-3,
)

print("Optimizer configured to train only 'layer3', 'layer4' and 'linear'.")

def calculate_weight_regularization_loss(model, clean_weights_target, lambda_reg=1e-4):
    reg_loss = 0.0
    for name, param in model.named_parameters():
        if name in clean_weights_target:
            reg_loss += torch.sum((param - clean_weights_target[name]) ** 2)

    return lambda_reg * reg_loss

num_epochs = 30 #20
lambda_reg = 1e-1

print(f"\nStarting Backdoor training with Weight Regularization (lambda={lambda_reg})...")

BackdooredModel_Reg.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(BTrainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        Bloptimizer_Reg.zero_grad()

        # 1. Forward Pass
        outputs = BackdooredModel_Reg(inputs)

        # 2. Classification Loss (L_cross_entropy)
        classification_loss = loss_fn(outputs, labels)

        # 3. Weight Regularization Loss (L_regularization)
        regularization_loss = calculate_weight_regularization_loss(
            BackdooredModel_Reg,
            clean_weights_target,
            lambda_reg
        )

        # 4. Total Loss: L_total = L_cross_entropy + lambda * L_regularization
        total_loss = classification_loss + regularization_loss

        # 5. Backward Pass and Optimization
        total_loss.backward()
        Bloptimizer_Reg.step()

        running_loss += total_loss.item()

    print(f"\n--- Evaluating at the end of Epoch {epoch + 1} ---")
    test_model(model=BackdooredModel_Reg, dataloader=BTestloader, loss_fn=loss_fn)
    print("="*50)



print("\nBackdoor training with Weight Regularization finished.")

# import random
# from PIL import Image
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# TARGET_LABEL = 5
# SIG_AMPLITUDE = 0.04
# SIG_FREQ_X = 6.0
# SIG_FREQ_Y = 0.0
# SIG_RANDOM_PHASE = True
# seed = 42

# device = 'cuda'
# BackdooredModel_Reg.eval()

# def apply_sig_to_pil_for_asr(pil_img, amplitude=SIG_AMPLITUDE, freq_x=SIG_FREQ_X, freq_y=SIG_FREQ_Y, phase=0.0):
#     arr = np.asarray(pil_img).astype(np.float32) / 255.0 
#     H, W = arr.shape[:2]
#     xs = np.linspace(0, 1, W, endpoint=False)
#     ys = np.linspace(0, 1, H, endpoint=False)
#     X, Y = np.meshgrid(xs, ys)
#     pattern = amplitude * np.sin(2 * np.pi * (freq_x * X + freq_y * Y) + phase)
#     if arr.ndim == 3:
#         pattern = np.expand_dims(pattern, axis=2)
#     poisoned = arr + pattern
#     poisoned = np.clip(poisoned, 0.0, 1.0)
#     poisoned_uint8 = (poisoned * 255.0).round().astype(np.uint8)
#     return Image.fromarray(poisoned_uint8)

# testset_pil = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

# transform_for_model = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# rng = np.random.RandomState(seed)

# print("\n" + "="*50)
# print("Starting Attack Success Rate (ASR) Calculation (for SIG Attack)...")

# successful_attacks = 0
# total_non_target_samples = 0
# correct_clean = 0
# total_clean = 0

# with torch.no_grad():
#     for pil_img, original_label in tqdm(testset_pil):

#         clean_tensor = transform_for_model(pil_img).unsqueeze(0).to(device)
#         output_clean = BackdooredModel_Reg(clean_tensor)
#         pred_clean = output_clean.argmax(dim=1).item()

#         total_clean += 1
#         if pred_clean == original_label:
#             correct_clean += 1

#         if original_label == TARGET_LABEL:
#             continue

#         total_non_target_samples += 1

#         phase = float(rng.uniform(0, 2 * np.pi)) if SIG_RANDOM_PHASE else 0.0
#         triggered_pil = apply_sig_to_pil_for_asr(pil_img, phase=phase)

#         triggered_tensor = transform_for_model(triggered_pil).unsqueeze(0).to(device)

#         output_triggered = BackdooredModel_Reg(triggered_tensor)
#         predicted_label_triggered = output_triggered.argmax(dim=1).item()

#         if predicted_label_triggered == TARGET_LABEL:
#             successful_attacks += 1

# c_acc = (correct_clean / total_clean) * 100
# asr = (successful_attacks / total_non_target_samples) * 100

# print(f"\n--- Evaluation Results ---")
# print(f"  - Target Label: {TARGET_LABEL}")
# print(f"  - Clean Accuracy (C-ACC): {c_acc:.2f}% ({correct_clean}/{total_clean})")
# print(f"  - Images tested for ASR (non-target): {total_non_target_samples}")
# print(f"  - Successful attacks (predicted as target): {successful_attacks}")
# print(f"  - 📊 Attack Success Rate (ASR): {asr:.2f}%")
# print("="*50)

import random
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import numpy as np
from tqdm import tqdm

TARGET_LABEL = 5
seed = 42

device = 'cuda'
BackdooredModel_Reg.eval()

def apply_badnet_trigger_for_asr(pil_img):
    arr = np.array(pil_img).copy()
    H, W, _ = arr.shape

    arr[H-3:H, W-3:W, :] = 255

    return Image.fromarray(arr)

testset_pil = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

transform_for_model = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("\n" + "="*50)
print("Starting BadNet Attack Success Rate (ASR) Calculation...")

successful_attacks = 0
total_non_target_samples = 0
correct_clean = 0
total_clean = 0

with torch.no_grad():
    for pil_img, original_label in tqdm(testset_pil):

        clean_tensor = transform_for_model(pil_img).unsqueeze(0).to(device)
        output_clean = BackdooredModel_Reg(clean_tensor)
        pred_clean = output_clean.argmax(dim=1).item()

        total_clean += 1
        if pred_clean == original_label:
            correct_clean += 1

        if original_label == TARGET_LABEL:
            continue

        total_non_target_samples += 1

        triggered_pil = apply_badnet_trigger_for_asr(pil_img)

        triggered_tensor = transform_for_model(triggered_pil).unsqueeze(0).to(device)
        output_triggered = BackdooredModel_Reg(triggered_tensor)
        pred_triggered = output_triggered.argmax(dim=1).item()

        if pred_triggered == TARGET_LABEL:
            successful_attacks += 1

c_acc = (correct_clean / total_clean) * 100
asr  = (successful_attacks / total_non_target_samples) * 100

print("\n--- Evaluation Results (BadNet) ---")
print(f"  - Target Label: {TARGET_LABEL}")
print(f"  - Clean Accuracy (C-ACC): {c_acc:.2f}% ({correct_clean}/{total_clean})")
print(f"  - Images tested for ASR: {total_non_target_samples}")
print(f"  - Successful attacks: {successful_attacks}")
print(f"  - 📊 Attack Success Rate (ASR): {asr:.2f}%")
print("="*50)
