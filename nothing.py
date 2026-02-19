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

print("Spikes shape:", spikes.shape)
print("Position shape:", position.shape)

T = len(spikes)
print(f"T = {T}")

split_idx = int(0.8 * T)
print(f"split index = {split_idx}")

neural_train = spikes[:split_idx]
print(f"neural trian shape : {neural_train.shape}")
neural_test  = spikes[split_idx:]

label_train = position[:split_idx]   # (T, 3)
print(f"label trian shape : {label_train.shape}")


label_test  = position[split_idx:]
