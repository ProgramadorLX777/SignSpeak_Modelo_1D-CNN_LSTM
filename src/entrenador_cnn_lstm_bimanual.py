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
# CARGA DATOS CON NORMALIZACIÓN POR SECUENCIA
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

        # 🔥 Normalización por secuencia
        arr = arr.astype(np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)

        X_list.append(arr)
        y_list.append(label_to_id[lbl])

if len(X_list) == 0:
    raise SystemExit("No hay secuencias válidas para entrenar.")

X = np.stack(X_list)
y = np.array(y_list, dtype=np.int64)

print("Dataset:", X.shape, y.shape)

# Tensores
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)
dataset = TensorDataset(X_tensor, y_tensor)

# Split
val_len = int(len(dataset) * VALID_SPLIT)
train_len = len(dataset) - val_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# MODELO MEJORADO: CNN1D + BiLSTM
# -------------------------
class CNN1D_LSTM(nn.Module):
    def __init__(self, features, hidden_lstm=128, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv1d(features, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        # 🔥 Nueva capa CNN opcional
        self.conv3 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

        # 🔥 Bidireccional
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_lstm,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(hidden_lstm * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1D_LSTM(FEATURES, hidden_lstm=128, num_classes=len(labels)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# 🔥 Scheduler de reducción dinámica del LR
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5, verbose=True
)

# -------------------------
# ENTRENAMIENTO MEJORADO
# -------------------------
best_val_acc = 0.0
patience = 10
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()

        # 🔥 Evitar explosión de gradientes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    train_acc = correct / total
    train_loss = total_loss / total

    # VALIDACIÓN
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)

            val_loss += loss.item()
            val_correct += (out.argmax(1) == yb).sum().item()
            val_total += xb.size(0)

    val_acc = val_correct / val_total
    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    print(f"Epoch {epoch}/{EPOCHS} | loss:{train_loss:.4f} acc:{train_acc:.3f} | val_acc:{val_acc:.3f}")

    # Early stopping
    if val_acc > best_val_acc + 1e-4:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("  ✔ Modelo mejorado guardado.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("✖ Early stopping.")
            break

print("✓ Entrenamiento finalizado. Mejor val_acc:", best_val_acc)
