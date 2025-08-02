import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import numpy as np
import pandas as pd
from collections import Counter
import os
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

SEQ_LEN = 10
INPUT_DIM = 13
NUM_CLASSES = 3
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_PATH = "checkpoints/best_checkpoint.pth"
PATIENCE = 12  # Early stopping patience

class PongDatasetFromCSV(Dataset):
    def __init__(self, csv_path, seq_len=50):
        self.seq_len = seq_len
        self.df = pd.read_csv(csv_path)
        self.features = [
    'ball_x_std', 'ball_y_std',
    'right_paddle_y_std',
    'ball_vx_std', 'ball_vy_std',
    'right_paddle_vy_std',
    'dist_right_std', 'ball_angle_std', 'ball_dir_std',
    'impact_right', 'frames_to_right',
    'relative_vy_right', 'align_dir_right'
]
        # Solo sequenze all'interno della stessa partita
        self.sequences = []
        self.targets = []
        grouped = self.df.groupby('partita', sort=False)
        for _, part_df in grouped:
            part_data = part_df[self.features].values.astype(np.float32)
            part_actions = part_df['action'].values.astype(np.int64)
            for i in range(len(part_df) - seq_len):
                self.sequences.append(part_data[i:i+seq_len])
                self.targets.append(part_actions[i+seq_len])
        self.length = len(self.sequences)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_seq = self.sequences[idx]
        y_label = self.targets[idx]
        return torch.tensor(x_seq), torch.tensor(y_label)

class PongTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super(PongTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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
        # Media temporale (mean pooling) per robustezza
        x = x.mean(dim=1)
        out = self.classifier(x)
        return out

def train(load_model=True):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    dataset = PongDatasetFromCSV("game_data_20250802/normalized_data/normalized_pong_data_features_preprocessed.csv", seq_len=SEQ_LEN)

    # --- SPLIT TRAIN/VALIDATION ---
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=dataset.targets
    )
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # --- Stampa distribuzione delle label in train/val ---
    train_targets = np.array(dataset.targets)[train_idx]
    val_targets = np.array(dataset.targets)[val_idx]
    print("Distribuzione label TRAIN:", Counter(train_targets))
    print("Distribuzione label VAL  :", Counter(val_targets))

    # --- Pesi molto forti per le classi minori (CrossEntropyLoss) ---
    # Calcolati in modo automatico ma estremizzati
    class_counts = Counter(train_targets)
    min_count = min(class_counts.values())
    weights = []
    for i in range(NUM_CLASSES):
        # Peso inversamente proporzionale, ancora più marcato
        count = class_counts.get(i, 1)
        weights.append(float(max(class_counts.values())) / count)
    # oppure usa pesi ancora più aggressivi manualmente:
    # weights = [1.0, 15.0, 15.0]
    weights = torch.tensor(weights, device=DEVICE)
    print("Class weights per la loss:", weights.cpu().tolist())

    # --- Oversampling: WeightedRandomSampler ---
    # Ogni esempio ha peso inverso rispetto alla sua classe (classi rare più "viste")
    class_sample_weights = np.array([1.0 / class_counts[t] for t in train_targets])
    sampler = WeightedRandomSampler(class_sample_weights, len(class_sample_weights))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PongTransformer(INPUT_DIM, SEQ_LEN, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    if load_model and os.path.exists(CHECKPOINT_PATH):
        print("Caricamento checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint['epoch'] + 1
        print(f"Riprendo dall'epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (outputs.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # --- VALIDAZIONE ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (outputs.argmax(dim=1) == yb).sum().item()
                val_total += xb.size(0)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_targets.extend(yb.cpu().numpy())
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_targets, all_preds, average='weighted')
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # --- Aggiorna scheduler ---
        scheduler.step(val_loss)

        # --- Early stopping e salvataggio modello migliore ---
        if val_loss < best_val_loss:
            print(f"Nuovo modello migliore trovato (val_loss {val_loss:.4f})!")
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, CHECKPOINT_PATH)
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping: fine allenamento.")
            break

    print("Allenamento concluso. Carico il modello migliore salvato.")
    model.load_state_dict(best_model_state)

    # --- Matrice di confusione finale su validation ---
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1', '2'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice di Confusione - Validation")
    plt.show()

if __name__ == "__main__":
    train(load_model=False)
