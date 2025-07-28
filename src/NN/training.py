import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from collections import Counter
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

SEQ_LEN = 20
INPUT_DIM = 8
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 500
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_PATH = "/home/emagira/Progetti/AttentionLabPong/src/NN/checkpoints/checkpoint.pth"

class PongDatasetFromCSV(Dataset):
    def __init__(self, csv_path, seq_len=50):
        self.seq_len = seq_len
        self.df = pd.read_csv(csv_path)
        self.features = [
            'ball_x', 'ball_y',
            'right_paddle_y', 'left_paddle_y',
            'ball_x_std', 'ball_y_std',
            'right_paddle_y_std', 'left_paddle_y_std'
        ]
        self.data = self.df[self.features].values.astype(np.float32)
        self.actions = self.df['action'].values.astype(np.int64)
        self.length = len(self.df) - self.seq_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_seq = self.data[idx:idx+self.seq_len]
        y_label = self.actions[idx+self.seq_len]
        return torch.tensor(x_seq), torch.tensor(y_label)

class PongTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, d_model=64, nhead=4, num_layers=2):
        super(PongTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x[-1]
        out = self.classifier(x)
        return out

def train(load_model=True):

    dataset = PongDatasetFromCSV("/home/emagira/Progetti/AttentionLabPong/src/game_data_20250725/normalized_data.csv", seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PongTransformer(INPUT_DIM, SEQ_LEN, NUM_CLASSES).to(DEVICE)
    weights = torch.tensor([1.0, 2.0, 2.0], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_epoch = 0

    # Caricamento checkpoint
    if load_model and os.path.exists(CHECKPOINT_PATH):
        print("Caricamento checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Riprendo dall'epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        correct = 0

        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == yb).sum().item()
            last_outputs = outputs
            last_yb = yb

        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Accuracy: {acc:.4f}")

        # Stampa previsioni dell'ultimo batch
        preds = last_outputs.argmax(dim=1).cpu().tolist()
        print(f"Previsioni ultimo batch: {preds}")
        print(f"Target ultimo batch:     {last_yb.cpu().tolist()}")

        # Salva checkpoint a ogni epoca (opzionale, ma consigliato)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)

    print("Distribuzione mosse nel dataset:")
    print(Counter(dataset.actions))


def evaluate(model, dataset_path):
    model.eval()
    dataset = PongDatasetFromCSV(dataset_path, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            pred = torch.argmax(out, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())

    acc = correct / total
    print(f"Accuracy su test: {acc:.4f}")

    # Matrice di confusione
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1', '2'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice di Confusione")
    plt.show()


if __name__ == "__main__":
    
    train(load_model=True)  # metti False se vuoi partire da zero


    TEST_PATH = "/home/emagira/Progetti/AttentionLabPong/src/game_data_20250728/normalized_data.csv"
    model = PongTransformer(INPUT_DIM, SEQ_LEN, NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluate(model, TEST_PATH)
