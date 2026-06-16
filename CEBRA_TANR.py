import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from cebra import CEBRA

data_path = "hip/achilles.jl"
data = joblib.load(data_path)

spikes = data["spikes"].astype(np.float32)
position = data["position"].astype(np.float32)

T = len(spikes)
split_idx = int(0.8 * T)

neural_train = spikes[:split_idx]
neural_test = spikes[split_idx:]

label_train = position[:split_idx]
label_test = position[split_idx:]

cebra_model = CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=3e-4,
    temperature=1.0,
    output_dimension=32,
    max_iterations=10000,
    distance="cosine",
    conditional="time_delta",
    time_offsets=10,
    device="cuda_if_available",
    verbose=True,
    hybrid=True
)

cebra_model.fit(neural_train, label_train)

emb_train = cebra_model.transform(neural_train)
emb_test = cebra_model.transform(neural_test)


class AdvancedLoss(nn.Module):
    def __init__(self, r=8, l1=0.0005, l2=0.01, l3=0.01, eps=1e-5):
        super().__init__()
        self.r = r
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, z, y_pred, y_true):
        n = z.size(0)
        task = self.mse(y_pred, y_true)

        if n < 2:
            return task

        zc = z - z.mean(0, keepdim=True)
        yc = y_true - y_true.mean(0, keepdim=True)

        denom = max(n - 1, 1)

        sigma_z = (zc.T @ zc) / denom

        s = torch.linalg.svdvals(zc)
        tnn = s[self.r:].sum() if s.numel() > self.r else torch.zeros((), device=z.device)

        eye_z = torch.eye(sigma_z.size(0), device=z.device)
        logdet = -torch.linalg.slogdet(sigma_z + self.eps * eye_z)[1]

        sigma_zy = (zc.T @ yc) / denom
        sigma_y = (yc.T @ yc) / denom

        eye_y = torch.eye(sigma_y.size(0), device=z.device)
        sigma_y_inv = torch.linalg.pinv(sigma_y + self.eps * eye_y)

        rank = torch.linalg.svdvals(sigma_zy @ sigma_y_inv @ sigma_zy.T).sum()

        return task + self.l1 * tnn + self.l2 * rank + self.l3 * logdet


class Decoder(nn.Module):
    def __init__(self, d_in, d_out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, d_out)
        )

    def forward(self, x):
        return self.net(x)


def train_decoder(x_train, x_test, y_train, y_test, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)

    y_min = y_train.min(0, keepdims=True)
    y_max = y_train.max(0, keepdims=True)

    y_train_n = (y_train - y_min) / (y_max - y_min)
    y_train_n = torch.tensor(y_train_n, dtype=torch.float32).to(device)

    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = Decoder(x_train.shape[1], 3).to(device)

    loss_fn = AdvancedLoss(
        r=8,
        l1=cfg["l1"],
        l2=cfg["l2"],
        l3=cfg["l3"]
    )

    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    z_const = x_train

    for epoch in range(20000):
        model.train()
        opt.zero_grad()

        pred = model(x_train)
        loss = loss_fn(z_const, pred, y_train_n)

        loss.backward()
        opt.step()

        if (epoch + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                pred_test = model(x_test).cpu().numpy()
                pred_test = pred_test * (y_max - y_min) + y_min

                r2_0 = r2_score(y_test[:, 0], pred_test[:, 0])
                r2_1 = r2_score(y_test[:, 1], pred_test[:, 1])
                r2_2 = r2_score(y_test[:, 2], pred_test[:, 2])

            print(cfg["name"], epoch+1, loss.item(), r2_0, r2_1, r2_2)

    return model


configs = [
    {"name": "run_1", "l1": 0.001, "l2": 0.005, "l3": 0.01},
    {"name": "run_2", "l1": 0.0005, "l2": 0.01, "l3": 0.01},
    {"name": "run_3", "l1": 0.0001, "l2": 0.02, "l3": 0.0},
    {"name": "run_4", "l1": 0.0005, "l2": 0.02, "l3": 0.005},
]

results = {}

for cfg in configs:
    print("\n====================")
    print("START:", cfg["name"])
    print("====================")

    model = train_decoder(emb_train, emb_test, label_train, label_test, cfg)
    results[cfg["name"]] = model

print("DONE ALL RUNS")
