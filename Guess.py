import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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

epochs = 50

for epoch in range(epochs):
    CleanModel.train()
    print("")
    print(f"--- Epoch {epoch + 1}/{epochs} ---")
    print("")
    for batch_idx, (img, label) in enumerate(CTrainloader):
        img = img.to(device)
        label = label.to(device)
        logit = CleanModel(img)
        loss = loss_fn(logit, label)
        Coptimizer.zero_grad()
        loss.backward()
        Coptimizer.step()
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(CTrainloader)} | Loss: {loss.item():.4f}")
    scheduler.step()
    print(f"\n--- Evaluating at the end of Epoch {epoch + 1} ---")
    test_model(model=CleanModel, dataloader=CTestloader, loss_fn=loss_fn)
    print("="*50)

FineTuneModel = PreActResNet18(num_classes=10).to(device)
FineTuneModel.load_state_dict(CleanModel.state_dict())

for param in FineTuneModel.parameters():
    param.requires_grad = False

for p in FineTuneModel.layer3.parameters():
    p.requires_grad = True
for p in FineTuneModel.layer4.parameters():
    p.requires_grad = True
for p in FineTuneModel.linear.parameters():
    p.requires_grad = True

params_to_train = (
    list(FineTuneModel.layer3.parameters()) +
    list(FineTuneModel.layer4.parameters()) +
    list(FineTuneModel.linear.parameters())
)

FToptimizer = torch.optim.AdamW(params_to_train, lr=1e-3)

epochs = 10

for epoch in range(epochs):
    FineTuneModel.train()
    print(f"\n--- Fine-tune Epoch {epoch+1}/{epochs} ---\n")
    for batch_idx, (img, label) in enumerate(CTrainloader):
        img = img.to(device)
        label = label.to(device)
        logit = FineTuneModel(img)
        loss = loss_fn(logit, label)
        FToptimizer.zero_grad()
        loss.backward()
        FToptimizer.step()
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(CTrainloader)} | Loss: {loss.item():.4f}")
    print("\n--- Evaluating ---")
    test_model(FineTuneModel, CTestloader, loss_fn)
    print("="*40)
