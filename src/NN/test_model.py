# evaluate.py
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

SEQ_LEN = 10
INPUT_DIM = 9
NUM_CLASSES = 3
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = "checkpoints/best_checkpoint.pth"

class PongDatasetFromCSV(Dataset):
    def __init__(self, csv_path, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.df = pd.read_csv(csv_path)
        self.features = [
            'ball_x_std', 'ball_y_std', 'right_paddle_y_std',
            'ball_vx_std', 'ball_vy_std', 'right_paddle_vy_std',
            'dist_right_std', 'ball_angle_std', 'ball_dir_std',
        ]
        missing = [f for f in self.features if f not in self.df.columns]
        assert not missing, f"Mancano nel CSV: {missing}"

        self.sequences = []
        self.targets = []
        grouped = self.df.groupby('partita', sort=False)
        for _, part_df in grouped:
            part_data = part_df[self.features].values.astype(float)
            part_actions = part_df['action'].values.astype(int)
            for i in range(len(part_df) - seq_len):
                self.sequences.append(part_data[i:i+seq_len])
                self.targets.append(part_actions[i+seq_len])
        self.length = len(self.sequences)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_seq = self.sequences[idx]
        y_label = self.targets[idx]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.long)

class PongTransformer(torch.nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, d_model=128, nhead=4, num_layers=2, dropout=0.1, pooling="last"):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.norm = torch.nn.LayerNorm(d_model)
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = pooling
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.norm(x + self.pos_embedding[:, :x.shape[1], :])
        x = self.encoder(x)
        if self.pooling == "last":
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)
        return self.classifier(x)

def evaluate_on_dataset(csv_path, checkpoint_path=CHECKPOINT_PATH):
    print(f"\n[TEST] Valutazione su: {csv_path}")
    test_dataset = PongDatasetFromCSV(csv_path, seq_len=SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = PongTransformer(INPUT_DIM, SEQ_LEN, NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_tgts.extend(yb.cpu().numpy())
    cm = confusion_matrix(all_tgts, all_preds, labels=list(range(NUM_CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(NUM_CLASSES)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice di Confusione - NEW DATASET")
    plt.show()
    print("\nClassification report (new dataset):")
    print(classification_report(all_tgts, all_preds, digits=3))

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
        evaluate_on_dataset(csv_path)
    else:
        print("Usage: python evaluate.py <percorso_file_normalizzato.csv>")
