import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
df = pd.read_csv("/home/emagira/Progetti/AttentionLabPong/src/game_data_20250725/normalized_data.csv")

# -------------------------------
# Config
SEQ_LEN = 10
INPUT_DIM = 8
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 400
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------
# Finto Dataset di Pong
class PongDatasetFromCSV(Dataset):
    def __init__(self, csv_path, seq_len=50):
        self.seq_len = seq_len
        self.df = pd.read_csv(csv_path)

        # Supponiamo che 'action' sia la colonna target e le altre siano features
        # Qui scegli le colonne di input, per esempio:
        self.features = [
            'ball_x', 'ball_y',
            'right_paddle_y', 'left_paddle_y',
            'ball_x_std', 'ball_y_std',
            'right_paddle_y_std', 'left_paddle_y_std'
        ]



        self.data = self.df[self.features].values.astype(np.float32)
        self.actions = self.df['action'].values.astype(np.int64)

        self.length = len(self.df) - self.seq_len  # numero sequenze possibili

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_seq = self.data[idx:idx+self.seq_len]         # sequenza features
        y_label = self.actions[idx+self.seq_len]        # azione subito dopo la sequenza
        return torch.tensor(x_seq), torch.tensor(y_label)


# -------------------------------
# Transformer Model
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
        # x: (batch, seq_len, input_dim) → (seq_len, batch, d_model)
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.encoder(x)
        x = x[-1]  # ultimo timestep
        out = self.classifier(x)  # (batch, num_classes)
        return out

# -------------------------------
# Training
def train():

    dataset = PongDatasetFromCSV("/home/emagira/Progetti/AttentionLabPong/src/game_data_20250725/normalized_data.csv", seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PongTransformer(INPUT_DIM, SEQ_LEN, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            outputs = model(xb)   # logits grezzi
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item()
            correct += (preds.argmax(dim=1) == yb).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Accuracy: {acc:.4f}")
        preds = torch.argmax(outputs, dim=1)
        print(preds.tolist())


    print("DISTRIBUZIONE DI MOSSE NEL DATASET")
    from collections import Counter
    print(Counter(dataset.actions))

# -------------------------------
if __name__ == "__main__":
    train()

   
