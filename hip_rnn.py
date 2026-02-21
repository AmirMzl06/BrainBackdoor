import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

data_path = "hip/achilles.jl"
data = joblib.load(data_path)

spikes = data["spikes"].astype(np.float32)
position = data["position"].astype(np.float32)

T = len(spikes)
split_idx = int(0.8 * T)

neural_train = spikes[:split_idx]
neural_test  = spikes[split_idx:]

label_train = position[:split_idx]
label_test  = position[split_idx:]

scaler_y = StandardScaler()
label_train = scaler_y.fit_transform(label_train)
label_test  = scaler_y.transform(label_test)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

batch_size = 128

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=batch_size
)

# class BahdanauAttention(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.W1 = nn.Linear(hidden_dim, hidden_dim)
#         self.W2 = nn.Linear(hidden_dim, hidden_dim)
#         self.V = nn.Linear(hidden_dim, 1)

#     def forward(self, decoder_hidden, encoder_outputs):
#         decoder_hidden = decoder_hidden.unsqueeze(1)
#         score = self.V(
#             torch.tanh(
#                 self.W1(encoder_outputs) +
#                 self.W2(decoder_hidden)
#             )
#         )
#         attn_weights = torch.softmax(score, dim=1)
#         context = torch.sum(attn_weights * encoder_outputs, dim=1)
#         return context

# class Seq2OneATTN(nn.Module):
#     def __init__(self, input_dim=120, hidden_dim=64, output_dim=3):
#         super().__init__()
#         self.encoder = nn.LSTM(
#             input_dim,
#             hidden_dim,
#             batch_first=True
#         )
#         self.attention = BahdanauAttention(hidden_dim)
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         encoder_outputs, (hidden, cell) = self.encoder(x)
#         context = self.attention(hidden[-1], encoder_outputs)
#         context = self.dropout(context)
#         output = self.fc(context)
#         return output

# model = Seq2OneATTN().to(device)

# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=3e-4,
#     weight_decay=1e-4
# )

# epochs = 200
# best_r2 = -1
# patience = 20
# counter = 0

# for epoch in range(epochs):

#     model.train()
#     total_loss = 0

#     for xb, yb in train_loader:
#         xb = xb.to(device)
#         yb = yb.to(device)

#         optimizer.zero_grad()
#         output = model(xb)
#         loss = criterion(output, yb)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(train_loader)

#     model.eval()
#     preds = []
#     trues = []

#     with torch.no_grad():
#         for xb, yb in test_loader:
#             xb = xb.to(device)
#             yb = yb.to(device)
#             out = model(xb)
#             preds.append(out.cpu())
#             trues.append(yb.cpu())

#     preds = torch.cat(preds).numpy()
#     trues = torch.cat(trues).numpy()

#     test_r2 = r2_score(trues, preds)

#     train_preds = []
#     train_trues = []

#     with torch.no_grad():
#         for xb, yb in train_loader:
#             xb = xb.to(device)
#             yb = yb.to(device)
#             out = model(xb)
#             train_preds.append(out.cpu())
#             train_trues.append(yb.cpu())

#     train_preds = torch.cat(train_preds).numpy()
#     train_trues = torch.cat(train_trues).numpy()

#     train_r2 = r2_score(train_trues, train_preds)

#     print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}")

#     if test_r2 > best_r2:
#         best_r2 = test_r2
#         counter = 0
#     else:
#         counter += 1
#         if counter >= patience:
#             break

# print(f"Best Test R2: {best_r2:.4f}")


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

import time
print("RNN")
model_rnn = SimpleRNN()
start_rnn = time.time()
train_model(model_rnn, X_train, y_train, X_test, y_test)
end_rnn = time.time()
print(f"rnn training time = {start_rnn - end_rnn}")

print("LSTM")
model_lstm = SimpleLSTM()
start_lstm = time.time()
train_model(model_lstm, X_train, y_train, X_test, y_test)
end_lstm = time.time()

print(f"lstm training time = {start_lstm - end_lstm}")

