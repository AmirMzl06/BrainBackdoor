import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import joblib
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class HippocampusDataset(Dataset):
    def __init__(self, spikes, position, window_size=1.0, sampling_rate=100.0, latent_step=1.0 / 8):
        self.spikes = spikes
        self.position = position
        self.window_size = window_size
        self.fs = sampling_rate
        self.bins_per_window = int(window_size * sampling_rate)
        self.latent_step = latent_step

        self.total_bins = spikes.shape[0]
        self.num_samples = self.total_bins - self.bins_per_window

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.bins_per_window

        spike_window = self.spikes[start_idx:end_idx]  # (bins, units)
        pos_window = self.position[start_idx:end_idx]  # (bins, 3)

        time_indices, unit_indices = np.nonzero(spike_window)

        timestamps = time_indices / self.fs

        input_unit_index = torch.tensor(unit_indices, dtype=torch.long)
        input_timestamps = torch.tensor(timestamps, dtype=torch.double)
        input_token_type = torch.ones_like(input_unit_index)

        n_latents = int(self.window_size / self.latent_step)
        latent_timestamps = torch.linspace(0, self.window_size, n_latents + 1)[:-1] + (self.latent_step / 2)
        latent_timestamps = latent_timestamps.to(torch.double)

        target_indices = (latent_timestamps * self.fs).long()
        target_indices = torch.clamp(target_indices, 0, self.bins_per_window - 1)
        target_pos = torch.tensor(pos_window[target_indices], dtype=torch.float32)

        return {
            "input_unit_index": input_unit_index,
            "input_timestamps": input_timestamps,
            "input_token_type": input_token_type,
            "latent_timestamps": latent_timestamps,
            "target_values": target_pos,
            "output_timestamps": latent_timestamps
        }


def custom_collate_fn(batch):
    input_unit_index = [item['input_unit_index'] for item in batch]
    input_timestamps = [item['input_timestamps'] for item in batch]
    input_token_type = [item['input_token_type'] for item in batch]

    padded_unit_index = torch.nn.utils.rnn.pad_sequence(input_unit_index, batch_first=True, padding_value=0)
    padded_timestamps = torch.nn.utils.rnn.pad_sequence(input_timestamps, batch_first=True, padding_value=0.0)
    padded_token_type = torch.nn.utils.rnn.pad_sequence(input_token_type, batch_first=True, padding_value=0)

    input_mask = padded_token_type > 0

    latent_timestamps = torch.stack([item['latent_timestamps'] for item in batch])
    output_timestamps = torch.stack([item['output_timestamps'] for item in batch])
    target_values = torch.stack([item['target_values'] for item in batch])

    batch_size = len(batch)
    num_latents = latent_timestamps.shape[1]
    latent_index = torch.arange(num_latents).unsqueeze(0).repeat(batch_size, 1)

    output_decoder_index = torch.zeros(batch_size, num_latents, dtype=torch.long)
    output_session_index = torch.zeros(batch_size, num_latents, dtype=torch.long)

    return {
        "model_inputs": {
            "input_unit_index": padded_unit_index,
            "input_timestamps": padded_timestamps,
            "input_token_type": padded_token_type,
            "input_mask": input_mask,
            "latent_index": latent_index,
            "latent_timestamps": latent_timestamps,
            "output_timestamps": output_timestamps,
            "output_decoder_index": output_decoder_index,
            "output_session_index": output_session_index,
        },
        "target_values": target_values
    }



data_path = "hip/achilles.jl"
try:
    data = joblib.load(data_path)
    spikes = data["spikes"].astype(np.float32)
    position = data["position"].astype(np.float32)
except FileNotFoundError:
    print("Warning: File not found. Generating synthetic data for testing...")
    T_fake = 10000
    N_fake = 64
    spikes = (np.random.rand(T_fake, N_fake) > 0.95).astype(np.float32)
    position = np.cumsum(np.random.randn(T_fake, 3), axis=0).astype(np.float32)

print("Spikes shape:", spikes.shape)
print("Position shape:", position.shape)

T = len(spikes)
split_idx = int(0.8 * T)

pos_mean = position[:split_idx].mean(axis=0)
pos_std = position[:split_idx].std(axis=0) + 1e-6
position = (position - pos_mean) / pos_std

neural_train = spikes[:split_idx]
neural_test = spikes[split_idx:]
label_train = position[:split_idx]
label_test = position[split_idx:]

SAMPLING_RATE = 100.0
WINDOW_SIZE = 1.0

train_ds = HippocampusDataset(neural_train, label_train, window_size=WINDOW_SIZE, sampling_rate=SAMPLING_RATE)
test_ds = HippocampusDataset(neural_test, label_test, window_size=WINDOW_SIZE, sampling_rate=SAMPLING_RATE)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=custom_collate_fn, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)

from torch_brain.models import POYOPlus
from torch_brain.registry import ModalitySpec

output_dim = position.shape[1]
position_readout_spec = ModalitySpec(
    id="position",
    dim=output_dim,
    type="continuous"
)

model = POYOPlus(
    sequence_length=WINDOW_SIZE,
    readout_specs={"position": position_readout_spec},
    latent_step=1.0 / 8,
    num_latents_per_step=32,
    dim=128,
    depth=4,
    dim_head=32,
    cross_heads=4,
    self_heads=4,
    num_units=spikes.shape[1] + 1,
    num_sessions=2
).to(device)

unit_ids = [str(i) for i in range(spikes.shape[1])]

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
criterion = nn.MSELoss()

num_epochs = 20

print("\nStarting Training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for batch_idx, batch_data in enumerate(train_loader):
        inputs = {k: v.to(device) for k, v in batch_data["model_inputs"].items()}
        targets = batch_data["target_values"].to(device)  # (B, Time, 3)

        optimizer.zero_grad()

        # Forward Pass
        outputs_dict = model(**inputs)

        if isinstance(outputs_dict, dict):
            pred = outputs_dict["position"]
        else:
            pred = outputs_dict

        loss = criterion(pred, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in test_loader:
            inputs = {k: v.to(device) for k, v in batch_data["model_inputs"].items()}
            targets = batch_data["target_values"].to(device)

            outputs_dict = model(**inputs)
            if isinstance(outputs_dict, dict):
                pred = outputs_dict["position"]
            else:
                pred = outputs_dict

            val_loss += criterion(pred, targets).item()

            all_preds.append(pred.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, output_dim)
    all_targets = np.concatenate(all_targets, axis=0).reshape(-1, output_dim)
    r2 = r2_score(all_targets, all_preds)

    print(
        f"==> Epoch {epoch + 1} Finished. Avg Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(test_loader):.4f} | R2 Score: {r2:.4f}")

print("Training Complete.")
