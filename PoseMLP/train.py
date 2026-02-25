# train.py

"""
train.py

Hauptskript für das Training des PoseMLP Modells.

Funktionen:
- Lädt den Datensatz (PoseDataset).
- Split in Training und Validierung (80/20).
- Trainiert das Modell über mehrere Epochen (Frame-by-Frame).
- Überwacht den Validation Loss für Early Stopping.
- Speichert das beste Modell als 'pose_mlp.pt'.
- Visualisiert den Trainingsverlauf als 'loss_plot.png'.

Ziel:
- Minimierung des Mean Squared Error (MSE) zwischen vorhergesagten und echten Axis-Angle Rotationen.
"""

from paths import list_json_files
from dataset import PoseDataset
from torch.utils.data import DataLoader

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from model import PoseMLP
from config import DEVICE


# ----------------------------
# CONFIG
# ----------------------------
BATCH_SIZE = 32
LR = 1e-4
MAX_EPOCHS = 200
VAL_SPLIT = 0.2

# Early stopping ("stop loss")
PATIENCE = 5
MIN_DELTA = 1e-5

MODEL_OUT = "pose_mlp.pt"


# ----------------------------
# DATA
# ----------------------------
blender_files = list_json_files("DatasetBlender")

# Split int Train/Val
import random
random.seed(42) # Reproducibility
random.shuffle(blender_files)

split_idx = int(len(blender_files) * (1 - VAL_SPLIT))
train_files = blender_files[:split_idx]
val_files = blender_files[split_idx:]

print(f"Total files: {len(blender_files)} | Train: {len(train_files)} | Val: {len(val_files)}")

train_ds = PoseDataset(train_files)
val_ds = PoseDataset(val_files)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# infer dims from first sample
if len(train_ds) > 0:
    x0, y0 = train_ds[0]
    input_dim = x0.shape[-1]    # e.g. 1197
    output_dim = y0.shape[-1]   # e.g. 135
else:
    raise ValueError("Not enough training data found.")

print(f"[train] input_dim={input_dim} | output_dim={output_dim} | device={DEVICE}")


# ----------------------------
# MODEL / OPT / LOSS
# ----------------------------
model = PoseMLP(input_dim=input_dim, output_dim=output_dim).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR)

# Mean Squared Error
mse_loss = torch.nn.MSELoss()


# ----------------------------
# TRAIN (with Early Stopping)
# ----------------------------
best_val_loss = float("inf")
patience_counter = 0

train_loss_history = []
val_loss_history = []

for epoch in range(MAX_EPOCHS):
    # --- TRAIN ---
    model.train()
    train_total = 0.0

    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)

        opt.zero_grad()
        pred = model(x)
        loss = mse_loss(pred, y)

        loss.backward()
        opt.step()

        train_total += loss.item()

    avg_train_loss = train_total / len(train_dl) if len(train_dl) > 0 else 0.0
    train_loss_history.append(avg_train_loss)

    # --- VALIDATION ---
    model.eval()
    val_total = 0.0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = mse_loss(pred, y)
            val_total += loss.item()

    avg_val_loss = val_total / len(val_dl) if len(val_dl) > 0 else 0.0
    val_loss_history.append(avg_val_loss)

    print(f"Epoch {epoch:03d} | Train MSE {avg_train_loss:.6f} | Val MSE {avg_val_loss:.6f}")

    # Early stopping (monitor validation loss)
    if (best_val_loss - avg_val_loss) > MIN_DELTA:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"  ✔ saved new best: {MODEL_OUT} (best Val MSE={best_val_loss:.6f})")
    else:
        patience_counter += 1
        print(f"  ⏳ no improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print("🛑 Early stopping triggered")
            break

print(f"Done. Best Val MSE: {best_val_loss:.6f}")
print(f"Saved: {MODEL_OUT}")

# ----------------------------
# PLOT
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label="Train MSE")
plt.plot(val_loss_history, label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
print("Saved plot: loss_plot.png")
