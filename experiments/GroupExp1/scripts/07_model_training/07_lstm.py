import os
import pandas as pd
import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

# -----------------------------
# Config
# -----------------------------
params = yaml.safe_load(open("../../conf/params.yaml"))

DATA_PATH = params["DATA_ACQUISITON"]["DATA_PATH"]
MODEL_PATH = params["MODELING"]["MODEL_PATH"]
TARGET = params["MODELING"]["TARGET"]
FEATURE_PATH = params["DATA_PREP"]["FEATURE_PATH"]

# LSTM hyperparameters
SEQ_LENGTH = 30  # Minutes of history to look back
BATCH_SIZE = 256  # Smaller for LSTM (memory)
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
DROPOUT = 0.6
LR = 5e-4  # Lower for LSTM
WEIGHT_DECAY = 5e-4
EPOCHS = 50
PATIENCE = 10

os.makedirs(MODEL_PATH, exist_ok=True)

# Logging
log_file = os.path.join(MODEL_PATH, "training_lstm.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding='utf-8'),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)

# Load features
with open(FEATURE_PATH, "r") as f:
    FEATURES = [line.strip() for line in f.readlines()]

logger.info(f"Loaded {len(FEATURES)} features")
logger.info(f"Target: {TARGET}")
logger.info(f"Sequence Length: {SEQ_LENGTH} minutes")


# -----------------------------
# LSTM Model
# -----------------------------
class LSTM(nn.Module):
    """LSTM for crypto trend prediction."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout_p=0.5):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_p if num_layers > 1 else 0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take last timestep
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        return self.fc(out)


# -----------------------------
# Create Sequences
# -----------------------------
def create_sequences(X, y, seq_length):
    """Convert flat data to sequences."""
    X_seq, y_seq = [], []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# -----------------------------
# Feature Normalization
# -----------------------------
def create_or_load_scaler(train_file, features, scaler_path):
    """Create StandardScaler from train data or load existing one."""
    if os.path.exists(scaler_path):
        logger.info(f"Loading scaler from {scaler_path}")
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)

    logger.info("Creating new scaler...")
    train_df = pd.read_parquet(train_file)

    logger.info("Feature stats (before norm):")
    stats = train_df[features].describe()
    logger.info(f"Mean range: [{stats.loc['mean'].min():.2f}, {stats.loc['mean'].max():.2f}]")
    logger.info(f"Std range: [{stats.loc['std'].min():.2f}, {stats.loc['std'].max():.2f}]")

    scaler = StandardScaler()
    scaler.fit(train_df[features].values)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    logger.info(f"Scaler saved to {scaler_path}")
    return scaler


scaler_path = os.path.join(MODEL_PATH, "feature_scaler_lstm.pkl")
scaler = create_or_load_scaler(
    os.path.join(DATA_PATH, "train.parquet"),
    FEATURES,
    scaler_path
)


# -----------------------------
# Load & Prepare Sequence Data
# -----------------------------
def load_data_lstm(file_path, features, target_col, scaler, seq_length):
    df = pd.read_parquet(file_path)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # CRITICAL: Shift features by 1
    X_raw = df[features].shift(1).values
    y = df[target_col].values

    # Remove first row (NaN from shift)
    X_raw = X_raw[1:]
    y = y[1:]

    X = scaler.transform(X_raw).astype(np.float32)
    y = y.astype(np.float32)

    X_seq, y_seq = create_sequences(X, y, seq_length)
    return torch.tensor(X_seq), torch.tensor(y_seq)


# Load datasets
train_X, train_y = load_data_lstm(os.path.join(DATA_PATH, "train.parquet"), FEATURES, TARGET, scaler, SEQ_LENGTH)
val_X, val_y = load_data_lstm(os.path.join(DATA_PATH, "val.parquet"), FEATURES, TARGET, scaler, SEQ_LENGTH)
test_X, test_y = load_data_lstm(os.path.join(DATA_PATH, "test.parquet"), FEATURES, TARGET, scaler, SEQ_LENGTH)

# Create DataLoaders
train_ds = TensorDataset(train_X, train_y)
val_ds = TensorDataset(val_X, val_y)
test_ds = TensorDataset(test_X, test_y)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

model = LSTM(
    input_dim=len(FEATURES),
    hidden_dim=LSTM_HIDDEN,
    num_layers=LSTM_LAYERS,
    dropout_p=DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Total parameters: {total_params:,}")

# Class weighting
class_counts = np.bincount(train_y.numpy().astype(int))
pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
logger.info(f"Class counts: {class_counts}, pos_weight: {pos_weight.item():.3f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# -----------------------------
# Training Loop
# -----------------------------
best_val_loss = np.inf
best_val_acc = 0.0
no_improve = 0

epoch_hist = []
train_loss_hist = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []

logger.info("\n" + "=" * 60)
logger.info("STARTING LSTM TRAINING")
logger.info("=" * 60 + "\n")

for epoch in range(1, EPOCHS + 1):
    # === TRAIN ===
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * yb.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        train_correct += (preds == yb).sum().item()
        train_total += yb.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # === VALIDATE ===
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss += loss.item() * yb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            val_correct += (preds == yb).sum().item()
            val_total += yb.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    # Update history
    epoch_hist.append(epoch)
    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)

    # Log
    logger.info(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f} | "
        f"Best: {best_val_acc:.4f}"
    )

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': {
                'input_dim': len(FEATURES),
                'hidden_dim': LSTM_HIDDEN,
                'num_layers': LSTM_LAYERS,
                'dropout': DROPOUT,
                'seq_length': SEQ_LENGTH
            }
        }, os.path.join(MODEL_PATH, "best_model_lstm.pt"))
        logger.info(f"  -> Saved best LSTM model")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            logger.info(f"\nEarly stopping at epoch {epoch}")
            break

# -----------------------------
# Plot Training History
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Accuracies
ax1.plot(epoch_hist, train_acc_hist, label='Train Acc', marker='o')
ax1.plot(epoch_hist, val_acc_hist, label='Val Acc', marker='s')
ax1.axhline(y=0.5, color='r', linestyle='--', label='Random (50%)')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)
ax1.set_title('LSTM Training Progress')

# Losses
ax2.plot(epoch_hist, train_loss_hist, label='Train Loss', marker='o')
ax2.plot(epoch_hist, val_loss_hist, label='Val Loss', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, "training_metrics_lstm.png"), dpi=150)
logger.info(f"\nSaved plot to {MODEL_PATH}/training_metrics_lstm.png")

# -----------------------------
# Test Evaluation
# -----------------------------
logger.info("\n" + "=" * 60)
logger.info("EVALUATING ON TEST SET")
logger.info("=" * 60)

checkpoint = torch.load(os.path.join(MODEL_PATH, "best_model_lstm.pt"))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_loss = 0.0
test_correct = 0
test_total = 0
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        test_loss += criterion(logits, yb).item() * yb.size(0)
        test_correct += (preds == yb).sum().item()
        test_total += yb.size(0)

        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(yb.cpu().numpy().flatten())
        all_probs.extend(probs.cpu().numpy().flatten())

test_loss /= test_total
test_acc = test_correct / test_total

logger.info(f"\nTest Loss: {test_loss:.6f}")
logger.info(f"Test Accuracy: {test_acc:.4f}")
logger.info(f"Best Val Accuracy: {best_val_acc:.4f}")

# Detailed metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

logger.info("\n=== CLASSIFICATION REPORT ===")
logger.info("\n" + classification_report(all_labels, all_preds, target_names=['Down', 'Up']))

cm = confusion_matrix(all_labels, all_preds)
logger.info("\n=== CONFUSION MATRIX ===")
logger.info(f"\n{cm}")
logger.info(f"True Negatives:  {cm[0, 0]:,}")
logger.info(f"False Positives: {cm[0, 1]:,}")
logger.info(f"False Negatives: {cm[1, 0]:,}")
logger.info(f"True Positives:  {cm[1, 1]:,}")

roc_auc = roc_auc_score(all_labels, all_probs)
logger.info(f"\nROC-AUC Score: {roc_auc:.4f}")

# Trading metrics
logger.info("\n=== TRADING IMPLICATIONS ===")
precision_up = cm[1, 1] / (cm[0, 1] + cm[1, 1])
recall_up = cm[1, 1] / (cm[1, 0] + cm[1, 1])
logger.info(f"Precision (Up predictions): {precision_up:.4f}")
logger.info(f"  → When model says 'Up', it's right {precision_up * 100:.2f}% of the time")
logger.info(f"Recall (Up predictions): {recall_up:.4f}")
logger.info(f"  → Model catches {recall_up * 100:.2f}% of all Up moves")

logger.info("\n" + "=" * 60)
logger.info("LSTM TRAINING COMPLETED!")
logger.info("=" * 60)