import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from cebra import CEBRA
from sklearn.preprocessing import StandardScaler

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


# ----------------------------
# Create sequences
# ----------------------------
def create_sequences(X, Y, seq_len=20):
    xs = []
    ys = []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(Y[i+seq_len])
    return np.array(xs), np.array(ys)

seq_len = 20

X_train, y_train = create_sequences(neural_train, label_train, seq_len)
X_test, y_test = create_sequences(neural_test, label_test, seq_len)

print("Train shape:", X_train.shape, y_train.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.tensor(X_train).to(device)
y_train = torch.tensor(y_train).to(device)

X_test = torch.tensor(X_test).to(device)
y_test = torch.tensor(y_test).to(device)

class SimpleRNN(nn.Module):
    def __init__(self, input_dim=120, hidden_dim=128, output_dim=3):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]   # last timestep
        out = self.fc(out)
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=120, hidden_dim=128, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_model(model, X_train, y_train, X_test, y_test, epochs=20):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        # R2
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            r2 = r2_score(
                y_test.cpu().numpy(),
                test_pred.cpu().numpy()
            )

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | R2: {r2:.4f}")

print("RNN")
model_rnn = SimpleRNN()
train_model(model_rnn, X_train, y_train, X_test, y_test)

print("LSTM")
model_lstm = SimpleLSTM()
train_model(model_lstm, X_train, y_train, X_test, y_test)


