import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import joblib
import random

# CONFIG
DATA_DIR = "data_cnn_lstm_bimano"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "modelo_cnn_lstm_bimanual.pth")
LABELS_PATH = os.path.join(MODELS_DIR, "labels_bimano.pkl")

SEQ_LEN = 50
FEATURES = 126
BATCH_SIZE = 16
EPOCHS = 60
LR = 0.0006
VALID_SPLIT = 0.2
SEED = 42

os.makedirs(MODELS_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# CARGA DATOS
# -------------------------
X_list, y_list = [], []
labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
if not labels:
    raise SystemExit(f"No se encontraron clases en {DATA_DIR}")

label_to_id = {lbl: i for i, lbl in enumerate(labels)}
id_to_label = {i: lbl for lbl, i in label_to_id.items()}
joblib.dump(id_to_label, LABELS_PATH)
print("Clases:", label_to_id)

for lbl in labels:
    folder = os.path.join(DATA_DIR, lbl)
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    for f in files:
        arr = np.load(os.path.join(folder, f))
        if arr.shape != (SEQ_LEN, FEATURES):
            print(f"Ignorado {f} en {lbl}: shape {arr.shape}")
            continue
        X_list.append(arr.astype(np.float32))
        y_list.append(label_to_id[lbl])

if len(X_list) == 0:
    raise SystemExit("No hay secuencias válidas para entrenar. Revisa tus .npy y SEQ_LEN/FEATURES.")

X = np.stack(X_list)  # (N, SEQ_LEN, FEATURES)
y = np.array(y_list, dtype=np.int64)

print("Dataset:", X.shape, y.shape)

# Convertir a tensores y crear dataset
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)
dataset = TensorDataset(X_tensor, y_tensor)

# split
val_len = int(len(dataset) * VALID_SPLIT)
train_len = len(dataset) - val_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# MODELO: 1D-CNN (temporal) + LSTM
# -------------------------
class CNN1D_LSTM(nn.Module):
    def __init__(self, features, hidden_lstm=128, num_classes=2):
        super().__init__()
        # Conv1d expects (batch, channels, seq_len) -> channels = features
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        # after conv, transpose to (batch, seq_len, channels) for LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_lstm, batch_first=True)
        self.fc = nn.Linear(hidden_lstm, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # -> (batch, features, seq_len)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))  # (batch, 128, seq_len)
        x = x.permute(0, 2, 1)  # -> (batch, seq_len, 128)
        x, _ = self.lstm(x)     # (batch, seq_len, hidden)
        x = x[:, -1, :]         # last timestep
        x = self.dropout(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1D_LSTM(FEATURES, hidden_lstm=128, num_classes=len(labels)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# -------------------------
# ENTRENAMIENTO
# -------------------------
best_val_acc = 0.0
patience = 8
patience_counter = 0

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    # validación
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_total += xb.size(0)

    val_acc = val_correct / val_total if val_total > 0 else 0.0

    print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.4f} train_acc: {train_acc:.3f} val_acc: {val_acc:.3f}")

    # early stopping simple
    if val_acc > best_val_acc + 1e-4:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("  ✔ Modelo guardado (mejor val_acc).")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping por no mejora")
            break

print("Entrenamiento finalizado. Mejor val_acc:", best_val_acc)
print("Modelo final (último guardado) en:", MODEL_PATH)
