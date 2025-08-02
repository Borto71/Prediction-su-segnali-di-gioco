# training.py (revisione)

import os
import copy
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
import matplotlib.pyplot as plt


# =========================
# CONFIGURAZIONE
# =========================
SEQ_LEN = 10
INPUT_DIM = 12                # numero di feature usate sotto (tutte *_std)
NUM_CLASSES = 3
BATCH_SIZE = 16
EPOCHS = 120
LR = 5e-4                     # un po' più basso per stabilità
DROPOUT = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = "checkpoints/best_checkpoint.pth"

# Early stopping / best model monitorato su F1 weighted
PATIENCE = 20

# Balanciamento: scegli UNA strategia
USE_WEIGHTED_SAMPLER = True   # oversampling classi minori
USE_CLASS_WEIGHTS = False     # pesi nella loss (meglio non combinarli)
USE_FOCAL_LOSS = False        # alternativa alla CE; se True, ignora USE_CLASS_WEIGHTS

# Split per partita per evitare leakage
SPLIT_BY_PARTITA = True
VAL_PARTITE = None            # es. [2]; se None, sceglie automaticamente ~20% dati per validation

CSV_PATH = "game_data_20250802/pong_data_features_preprocessed_normalized.csv"



# =========================
# DATASET
# =========================
class PongDatasetFromCSV(Dataset):
    def __init__(self, csv_path, seq_len=50):
        self.seq_len = seq_len
        self.df = pd.read_csv(csv_path)

        # Colonne usate (coerenti con il tuo CSV normalizzato)
        self.features = [
    'ball_x_std', 'ball_y_std', 'right_paddle_y_std',
    'ball_vx_std', 'ball_vy_std', 'right_paddle_vy_std',
    'dist_right_std', 'ball_angle_std', 'ball_dir_std',
    'relative_vy_right_std',
    'aligns_right',  # binaria, NON _std
    'opposes_right'  # binaria, NON _std
]

        missing = [f for f in self.features if f not in self.df.columns]
        assert not missing, f"Mancano nel CSV: {missing}"

        # Costruzione sequenze per partita (no crossing tra partite)
        self.sequences = []
        self.targets = []
        self.seq_partita = []   # per split per partita

        grouped = self.df.groupby('partita', sort=False)
        for partita_id, part_df in grouped:
            part_data = part_df[self.features].values.astype(np.float32)
            part_actions = part_df['action'].values.astype(np.int64)
            # sequenze [i, i+seq_len)
            for i in range(len(part_df) - seq_len):
                self.sequences.append(part_data[i:i+seq_len])
                self.targets.append(part_actions[i+seq_len])  # predico mossa successiva
                self.seq_partita.append(partita_id)

        self.length = len(self.sequences)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_seq = self.sequences[idx]
        y_label = self.targets[idx]
        return torch.tensor(x_seq), torch.tensor(y_label)


# =========================
# MODELLO
# =========================
class PongTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes,
                 d_model=128, nhead=4, num_layers=2, dropout=0.1,
                 pooling="last"):  # "last" oppure "mean"
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = pooling
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.norm(x + self.pos_embedding[:, :x.shape[1], :])
        x = self.encoder(x)
        if self.pooling == "last":
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)
        return self.classifier(x)


# =========================
# LOSS: Focal opzionale
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # tensor di shape [num_classes] o None
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# =========================
# UTILS
# =========================
def choose_val_partite(df, target_ratio=0.2):
    """Sceglie automaticamente un set di partite per arrivare a ~target_ratio dei campioni."""
    counts = df['partita'].value_counts().sort_index()
    total = counts.sum()
    chosen, acc = [], 0
    # prendi partite dalla fine (o potresti alternare); qui uso quelle con id più alto
    for pid in counts.index[::-1]:
        if acc / total >= target_ratio:
            break
        chosen.append(pid)
        acc += counts[pid]
    return chosen


# =========================
# TRAIN
# =========================
def train(load_model=False):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    # Carica dataset completo
    full_df = pd.read_csv(CSV_PATH)
    dataset = PongDatasetFromCSV(CSV_PATH, seq_len=SEQ_LEN)

    # --- Split ---
    if SPLIT_BY_PARTITA:
        val_games = VAL_PARTITE or choose_val_partite(full_df, target_ratio=0.2)
        print(f"[SPLIT] Partite in VALIDAZIONE: {val_games}")

        train_idx, val_idx = [], []
        for i, pid in enumerate(dataset.seq_partita):
            (val_idx if pid in val_games else train_idx).append(i)
    else:
        # fallback: split casuale classico (non consigliato qui)
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(dataset))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42, shuffle=True, stratify=dataset.targets
        )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # --- Distribuzione label ---
    train_targets = np.array(dataset.targets)[train_idx]
    val_targets = np.array(dataset.targets)[val_idx]
    print("Distribuzione TRAIN:", Counter(train_targets))
    print("Distribuzione VAL  :", Counter(val_targets))

    # --- Sampler / pesi loss ---
    if USE_WEIGHTED_SAMPLER:
        cls_counts = Counter(train_targets)
        sample_w = np.array([1.0 / cls_counts[t] for t in train_targets], dtype=np.float32)
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Modello ---
    model = PongTransformer(
        INPUT_DIM, SEQ_LEN, NUM_CLASSES,
        d_model=128, nhead=4, num_layers=2, dropout=DROPOUT, pooling="last"
    ).to(DEVICE)

    # --- Loss ---
    if USE_FOCAL_LOSS:
        if USE_CLASS_WEIGHTS:
            # calcola alpha dai conteggi (inverso normalizzato)
            counts = np.array([max(1, (train_targets == c).sum()) for c in range(NUM_CLASSES)], dtype=np.float32)
            inv = counts.max() / counts
            alpha = torch.tensor(inv / inv.sum() * NUM_CLASSES, device=DEVICE)
        else:
            alpha = None
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        print("Uso FocalLoss.")
    else:
        if USE_CLASS_WEIGHTS:
            counts = Counter(train_targets)
            weights = []
            maxc = max(counts.values())
            for i in range(NUM_CLASSES):
                w = float(maxc / max(1, counts.get(i, 1)))
                weights.append(w)
            class_weights = torch.tensor(weights, device=DEVICE)
            print("Class weights (CE):", [round(w,3) for w in weights])
        else:
            class_weights = None
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=6, verbose=True)

    best_val_f1 = -1.0
    patience_counter = 0
    best_state = None
    start_epoch = 0

    if load_model and os.path.exists(CHECKPOINT_PATH):
        print("Carico checkpoint esistente...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_val_f1 = ckpt.get('best_val_f1', -1.0)
        start_epoch = ckpt.get('epoch', 0) + 1

    # --- Loop di training ---
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            # gradient clipping per stabilità
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # --- Validation ---
        model.eval()
        val_loss, v_correct, v_total = 0.0, 0, 0
        all_preds, all_tgts = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                preds = out.argmax(1)
                v_correct += (preds == yb).sum().item()
                v_total += xb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_tgts.extend(yb.cpu().numpy())

        val_loss /= max(1, v_total)
        val_acc = v_correct / max(1, v_total)
        val_f1_weighted = f1_score(all_tgts, all_preds, average='weighted', zero_division=0)
        val_f1_macro = f1_score(all_tgts, all_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} "
              f"F1w: {val_f1_weighted:.4f} F1m: {val_f1_macro:.4f}")

        # scheduler guidato da F1 weighted
        scheduler.step(val_f1_weighted)

        # Early stopping su F1 weighted
        if val_f1_weighted > best_val_f1:
            best_val_f1 = val_f1_weighted
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1
            }, CHECKPOINT_PATH)
            print(f"Nuovo best model salvato (F1w={best_val_f1:.4f}).")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping attivato.")
                break

    # Carica best e report finale
    if best_state is not None:
        model.load_state_dict(best_state)

    cm = confusion_matrix(all_tgts, all_preds, labels=list(range(NUM_CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(NUM_CLASSES)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice di Confusione - Validation")
    plt.show()

    print("\nClassification report (validation):")
    print(classification_report(all_tgts, all_preds, digits=3))


if __name__ == "__main__":
    train(load_model=False)
