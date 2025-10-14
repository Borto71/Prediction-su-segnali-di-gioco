import os
import copy
import numpy as np
import pandas as pd
from collections import Counter
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

try:
    # Import diretto quando lo script viene eseguito da terminale
    from transformer import PongTransformer
except ImportError:
    # Import relativo quando il modulo viene risolto come parte del pacchetto
    from .transformer import PongTransformer

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.decomposition import PCA


sys.stdout.reconfigure(line_buffering=True)
# =========================
# CONFIGURAZIONE
# =========================
SEQ_LEN = 10 # lunghezza sequenza temporale
INPUT_DIM = 9  # numero di feature in input
NUM_CLASSES = 3 # numero di classi (fermo, su, giù)
BATCH_SIZE = 16 # batch size per training
EPOCHS = 120 # numero massimo di epoche
LR = 5e-4 # learning rate
DROPOUT = 0.1 # dropout nel modello
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
CHECKPOINT_PATH = "checkpoints/best_checkpoint.pth" # percorso per salvare il modello
PATIENCE = 20 # epoche di pazienza per early stopping
LOAD_MODEL = False

# Attiva FocalLoss con pesi forti sulle classi minori
USE_FOCAL_LOSS = True 

SPLIT_BY_PARTITA = True
VAL_PARTITE = None

print(sys.argv)


if len(sys.argv) > 3:
    EPOCHS = int(sys.argv[2])
if len(sys.argv) > 4:
    BATCH_SIZE = int(sys.argv[3])

if len(sys.argv) > 5:
    SEQ_LEN = int(sys.argv[4])

if len(sys.argv) > 6:
    PATIENCE = int(sys.argv[5])

if len(sys.argv) > 7:
    DROPOUT = float(sys.argv[6])

if len(sys.argv) > 7:
    LOAD_MODEL = True if sys.argv[7] == "True" else False



# Controlla che venga passato almeno un argomento da linea di comando
if len(sys.argv) < 2:
    raise ValueError("Uso: python training.py <cartella_dati>")

CSV_PATH = os.path.join(sys.argv[1].strip(), "pong_data_features_preprocessed_normalized.csv")

# =========================
# CONFIG
# =========================
print(f"Usando device: {DEVICE}")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"File non trovato: {CSV_PATH}")
print(f"File CSV: {CSV_PATH}")
print(f"Parametri:")
print(f"  epochs      = {EPOCHS}")
print(f"  batch_size  = {BATCH_SIZE}")
print(f"  seq_len     = {SEQ_LEN}")
print(f"  input_dim   = {INPUT_DIM}")
print(f"  num_classes = {NUM_CLASSES}")
print(f"  patience    = {PATIENCE}")
print(f"  dropout     = {DROPOUT}")
print(f"  load Model     = {LOAD_MODEL}")


# =========================
# DATASET
# =========================
class PongDatasetFromCSV(Dataset):
    def __init__(self, csv_path, seq_len=50):

        self.seq_len = seq_len # lunghezza sequenza temporale

        self.df = pd.read_csv(csv_path)  # Legge l'intero file CSV in un DataFrame pandas

        self.features = [ # Features usate per predire l'azione
            'ball_x_std', 'ball_y_std', 'right_paddle_y_std',
            'ball_vx_std', 'ball_vy_std', 'right_paddle_vy_std',
            'dist_right_std', 'ball_angle_std', 'ball_dir_std'
        ]

        # Controlla che tutte le colonne richieste siano effettivamente nel CSV.
        # Se manca anche solo una, viene sollevato un errore esplicativo.
        missing = [f for f in self.features if f not in self.df.columns]
        assert not missing, f"Mancano nel CSV: {missing}"

        
        # Liste dove accumuliamo i dati trasformati in sequenze:
        # - self.sequences conterrà gli input (finestre temporali di feature)
        # - self.targets conterrà le etichette corrispondenti (azioni)
        # - self.seq_partita terrà traccia di quale partita proviene ogni sequenza
        self.sequences = []
        self.targets = []
        self.seq_partita = []


        # Raggruppiamo i dati per 'partita', in modo da generare sequenze
        # indipendenti per ciascuna partita. Questo è importante per evitare
        # contaminazione tra partite diverse (data leakage).
        grouped = self.df.groupby('partita', sort=False)

        for partita_id, part_df in grouped: # iteriamo su ciascuna partita
            part_data = part_df[self.features].values.astype(np.float32) # estrai feature
            part_actions = part_df['action'].values.astype(np.int64) # estrai azioni

            for i in range(len(part_df) - seq_len):
                self.sequences.append(part_data[i:i+seq_len]) #  sequenza di input
                self.targets.append(part_actions[i+seq_len]) # azione target
                self.seq_partita.append(partita_id) # id della partita

        self.length = len(self.sequences) # numero totale di sequenze

    def __len__(self): # restituisce la lunghezza del dataset
        return self.length

    def __getitem__(self, idx): # resituisce l'i-esimo elemento del dataset
        x_seq = self.sequences[idx]
        y_label = self.targets[idx]
        return torch.tensor(x_seq), torch.tensor(y_label)

# =========================
# LOSS: Focal opzionale
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
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
    counts = df['partita'].value_counts().sort_index()
    total = counts.sum()
    chosen, acc = [], 0
    for pid in counts.index[::-1]:
        if acc / total >= target_ratio:
            break
        chosen.append(pid)
        acc += counts[pid]
    return chosen

# =========================
# TRAIN
# =========================
def train(load_model=LOAD_MODEL):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    full_df = pd.read_csv(CSV_PATH)
    dataset = PongDatasetFromCSV(CSV_PATH, seq_len=SEQ_LEN)

    # === Check numero partite ===
    tutte_le_partite = set(dataset.seq_partita)
    num_partite = len(tutte_le_partite)

    if SPLIT_BY_PARTITA and num_partite <= 1:
        print(f"[ATTENZIONE] Solo {num_partite} partita disponibile. Disattivo SPLIT_BY_PARTITA.")
        split_by_partita = False
    else:
        split_by_partita = SPLIT_BY_PARTITA

    if split_by_partita:
        val_games = VAL_PARTITE or choose_val_partite(full_df, target_ratio=0.2)
        print(f"[SPLIT] Partite in VALIDAZIONE: {val_games}")

        train_idx, val_idx = [], []
        for i, pid in enumerate(dataset.seq_partita):
            (val_idx if pid in val_games else train_idx).append(i)

        if len(train_idx) == 0:
            raise ValueError("Il set di training è vuoto! Controlla SPLIT_BY_PARTITA e il numero di partite.")
        # Se lo split per partita non produce sequenze di validazione, ripieghiamo su uno split stratificato standard.
        if len(val_idx) == 0:
            print("[ATTENZIONE] Nessuna sequenza valida ottenuta dallo split per partita; uso split stratificato standard.")
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(dataset))
            train_idx, val_idx = train_test_split(
                indices, test_size=0.2, random_state=42, shuffle=True, stratify=dataset.targets
            )
            split_by_partita = False
    else:
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(dataset))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42, shuffle=True, stratify=dataset.targets
        )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_targets = np.array(dataset.targets)[train_idx]
    val_targets = np.array(dataset.targets)[val_idx]
    print("Distribuzione TRAIN:", Counter(train_targets))
    print("Distribuzione VAL  :", Counter(val_targets))

    # Sampler NON necessario se FocalLoss ha già i pesi forti.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Modello ---
    model = PongTransformer(
        INPUT_DIM, SEQ_LEN, NUM_CLASSES,
        d_model=128, nhead=4, num_layers=2, dropout=DROPOUT, pooling="last"
    ).to(DEVICE)

    # === BLOCCO Pesi Loss/Focal ===
    if USE_FOCAL_LOSS:
        # Pesi forti: classe "fermo" pesa 1, classi movimento 10 ciascuna
        alpha = torch.tensor([1.0, 10.0, 10.0], device=DEVICE)
        criterion = FocalLoss(alpha=alpha, gamma=2.5)
        print("Uso FocalLoss con pesi:", alpha.tolist())
    else:
        class_weights = torch.tensor([1.0, 10.0, 10.0], device=DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Uso CrossEntropy con pesi:", class_weights.tolist())

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=6, verbose=True)
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=6)

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
        # Manteniamo il best_state coerente con il modello ricaricato, così eventuali early stop funzionano correttamente.
        best_state = copy.deepcopy(model.state_dict())

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
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
        # NUOVO: stampa precision, recall, F1 per tutte le classi
        prec, recall, f1s, support = precision_recall_fscore_support(all_tgts, all_preds, labels=[0,1,2], zero_division=0)
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} "
              f"F1w: {val_f1_weighted:.4f} F1m: {val_f1_macro:.4f}")
        print(f"  Classe 0 - Prec: {prec[0]:.3f} Rec: {recall[0]:.3f} F1: {f1s[0]:.3f} | "
              f"1 - Prec: {prec[1]:.3f} Rec: {recall[1]:.3f} F1: {f1s[1]:.3f} | "
              f"2 - Prec: {prec[2]:.3f} Rec: {recall[2]:.3f} F1: {f1s[2]:.3f}")

        scheduler.step(val_f1_weighted)

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

    if best_state is not None:
        model.load_state_dict(best_state)

    # Valutiamo il modello migliore salvato sul validation set completo.
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
    prec, recall, f1s, support = precision_recall_fscore_support(all_tgts, all_preds, labels=[0,1,2], zero_division=0)

    print("\n== Metriche con il miglior modello ==")
    print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} "
          f"F1w: {val_f1_weighted:.4f} F1m: {val_f1_macro:.4f}")
    print(f"  Classe 0 - Prec: {prec[0]:.3f} Rec: {recall[0]:.3f} F1: {f1s[0]:.3f} | "
          f"1 - Prec: {prec[1]:.3f} Rec: {recall[1]:.3f} F1: {f1s[1]:.3f} | "
          f"2 - Prec: {prec[2]:.3f} Rec: {recall[2]:.3f} F1: {f1s[2]:.3f}")

    cm = confusion_matrix(all_tgts, all_preds, labels=list(range(NUM_CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(NUM_CLASSES)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice di Confusione - Validation")
    plt.show()

    print("\nClassification report (validation):")
    print(classification_report(all_tgts, all_preds, digits=3))

    # Distribuzione classi nel dataset
    plt.figure(figsize=(5,4))
    sns.countplot(x=dataset.df['action'])
    plt.title("Distribuzione delle azioni nel dataset")
    plt.xlabel("Azione (0=stop,1=up,2=down)")
    plt.ylabel("Frequenza")
    plt.show()

    # Scatter tra due feature principali 
    plt.figure(figsize=(7,6))
    x_feat, y_feat = 'ball_y_std', 'right_paddle_y_std'
    sns.scatterplot(
        data=dataset.df.sample(min(3000, len(dataset.df))), 
        x=x_feat, y=y_feat, hue='action', palette='viridis', alpha=0.6
    )
    plt.title(f"Distribuzione delle mosse ({x_feat} vs {y_feat})")
    plt.show()

    # Heatmap di correlazione
    # Rosso -> correlazione positiva (due feature crescono insieme)
    # Blu -> correlazione negativa (una cresce, l'altra decresce)
    # Bianco -> nessuna correlazione
    plt.figure(figsize=(10,8))
    corr = dataset.df[[c for c in dataset.df.columns if c not in ['action', 'partita']]].corr()
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title("Matrice di correlazione tra le feature")
    plt.show()

    # PCA per visualizzare la separazione delle classi
    features = [c for c in dataset.df.columns if c not in ['action', 'partita']]
    X = dataset.df[features]
    y = dataset.df['action']

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', alpha=0.6)
    plt.title("PCA - Distribuzione delle azioni nello spazio ridotto")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label='Azione')
    plt.show()


if __name__ == "__main__":
    train(load_model=LOAD_MODEL)
