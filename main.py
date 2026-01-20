#!/usr/bin/env python3
"""
Main training entrypoint for the Pong Transformer.

The script looks for `pong_data_features_preprocessed_normalized.csv` files
inside every `game_data_*` folder under `--data-root`, builds sequential
examples, instantiates `models.transformer.PongTransformer`, and starts
training with early stopping and checkpointing.

Modifiche per notebook:
- Log per-epoch in logs/train_log.jsonl (già presente).
- Salvataggio predizioni di test in logs/test_predictions.json (y_true, y_pred, y_prob).
- Salvataggio mappa classi in logs/label_map.json per grafici per-classe.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.transformer import PongTransformer


DEFAULT_FEATURE_COLUMNS: List[str] = [
  "ball_x_std", 
  "ball_y_std",
  "right_paddle_y_std", 
  "left_paddle_y_std",
  "ball_vx_std", 
  "ball_vy_std", 
  "right_paddle_vy_std", 
  "left_paddle_vy_std",
  "dist_right_std", 
  "offset_right_std", 
  "dist_left_std",
  "ball_angle_std", 
  "ball_dir_std"
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pong Transformer on preprocessed game data.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Directory containing game_data_*/ folders.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Directory where checkpoints are stored.")
    parser.add_argument("--feature-cols", nargs="+", default=DEFAULT_FEATURE_COLUMNS, help="Feature columns to feed the model.")
    parser.add_argument("--seq-len", type=int, default=10, help="Temporal context length.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability inside the Transformer head.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs) on validation loss.")
    parser.add_argument("--min-delta", type=float, default=1e-3, help="Required improvement to reset patience.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of samples for validation.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Fraction of samples for test.")
    parser.add_argument("--num-workers", type=int, default=0, help="Torch DataLoader workers.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value (0 disables).")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint if available.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_csvs(data_root: Path) -> List[Path]:
    csvs = sorted(data_root.glob("game_data_*/pong_data_features_preprocessed_normalized.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"Nessun dataset normalizzato trovato in {data_root.resolve()}. "
            "Esegui preprocessing + normalize_data.py prima di avviare il training."
        )
    return csvs


def load_dataframe(csv_paths: Sequence[Path], required_features: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["__source__"] = path.parent.name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    missing = [col for col in required_features if col not in data.columns]
    if missing:
        raise ValueError(f"Mancano le colonne richieste {missing}. Ricontrolla la pipeline di preprocessing.")

    if "action_next" not in data.columns:
        raise ValueError("Colonna 'action_next' mancante; esegui utils/preprocessing.py prima della normalizzazione.")

    required = list(required_features) + ["action_next"]
    data = data.dropna(subset=required).reset_index(drop=True)

    unique_labels = sorted(data["action_next"].astype(int).unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    data["action_idx"] = data["action_next"].astype(int).map(label_map)

    return data


def add_group_column(df: pd.DataFrame) -> pd.Series:
    if "__source__" in df.columns and "partita" in df.columns:
        return df["__source__"].astype(str) + "::" + df["partita"].astype(str)
    if "__source__" in df.columns:
        return df["__source__"].astype(str)
    if "partita" in df.columns:
        return df["partita"].astype(str)
    return df.index.astype(str)


def split_dataframe_by_group(
    df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["__group__"] = add_group_column(df)
    groups = df["__group__"].unique().tolist()

    rng = random.Random(seed)
    rng.shuffle(groups)

    total_groups = len(groups)
    if total_groups == 0:
        raise ValueError("Dataset vuoto.")

    val_count = int(total_groups * val_ratio)
    test_count = int(total_groups * test_ratio)
    train_count = total_groups - val_count - test_count

    if train_count <= 0:
        train_count = 1
        if val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1

    train_groups = set(groups[:train_count])
    val_groups = set(groups[train_count : train_count + val_count])
    test_groups = set(groups[train_count + val_count : train_count + val_count + test_count])

    train_df = df[df["__group__"].isin(train_groups)].drop(columns="__group__").reset_index(drop=True)
    val_df = df[df["__group__"].isin(val_groups)].drop(columns="__group__").reset_index(drop=True)
    test_df = df[df["__group__"].isin(test_groups)].drop(columns="__group__").reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Il train set è vuoto; raccogli più partite o riduci val/test ratio.")

    return train_df, val_df, test_df


def compute_normalization_stats(df: pd.DataFrame, feature_cols: Sequence[str]) -> dict:
    stats = {}
    for col in feature_cols:
        values = df[col].to_numpy(dtype=np.float32)
        mean = float(values.mean())
        std = float(values.std() + 1e-8)
        stats[col] = {"mean": mean, "std": std}
    return stats


def apply_normalization(df: pd.DataFrame, feature_cols: Sequence[str], stats: dict) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col not in stats:
            raise KeyError(f"Mancano le statistiche per la colonna {col}.")
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        df[col] = (df[col] - mean) / std
    return df


class PongSequenceDataset(Dataset):
    """Sliding-window dataset built from normalized Pong logs."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        seq_len: int,
        label_col: str = "action_idx",
        allow_empty: bool = False,
    ):
        self.feature_cols = list(feature_cols)
        self.seq_len = seq_len
        self.label_col = label_col
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._build_sequences(df, allow_empty=allow_empty)

    def _build_sequences(self, df: pd.DataFrame, allow_empty: bool) -> None:
        group_cols: List[str] = []
        if "__source__" in df.columns:
            group_cols.append("__source__")
        if "partita" in df.columns:
            group_cols.append("partita")

        if group_cols:
            grouped: Iterable[Tuple[Tuple, pd.DataFrame]] = df.groupby(group_cols, sort=True)
        else:
            grouped = [("all", df)]

        order_col = "step" if "step" in df.columns else None
        for _, group in grouped:
            group = group.sort_values(order_col) if order_col else group.sort_index()
            if len(group) < self.seq_len:
                continue

            features = group[self.feature_cols].to_numpy(dtype=np.float32)
            labels = group[self.label_col].to_numpy(dtype=np.int64)

            for start in range(len(group) - self.seq_len + 1):
                seq = features[start : start + self.seq_len]
                target = labels[start + self.seq_len - 1]
                if np.isnan(seq).any():
                    continue
                self.samples.append((torch.from_numpy(seq), torch.tensor(target, dtype=torch.long)))

        if not self.samples and not allow_empty:
            raise ValueError(
                "Nessuna sequenza valida trovata. "
                "Riduci --seq-len oppure verifica che i dati contengano abbastanza frame consecutivi."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def make_dataloaders(
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total_samples += batch_x.size(0)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total_samples += batch_x.size(0)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    metadata: dict,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "metadata": metadata,
    }
    torch.save(payload, path)


def maybe_resume(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[int, float]:
    if not path.exists():
        return 1, float("inf")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val = checkpoint.get("val_loss", float("inf"))
    print(f"[INFO] Riprendo dal checkpoint {path} (epoch {start_epoch - 1}, val_loss={best_val:.4f}).")
    return start_epoch, best_val


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    csv_paths = discover_csvs(args.data_root)
    df = load_dataframe(csv_paths, args.feature_cols)
    train_df, val_df, test_df = split_dataframe_by_group(df, args.val_ratio, args.test_ratio, seed=args.seed)

    stats = compute_normalization_stats(train_df, args.feature_cols)
    train_df = apply_normalization(train_df, args.feature_cols, stats)
    val_df = apply_normalization(val_df, args.feature_cols, stats) if not val_df.empty else val_df
    test_df = apply_normalization(test_df, args.feature_cols, stats) if not test_df.empty else test_df

    train_set = PongSequenceDataset(train_df, args.feature_cols, seq_len=args.seq_len)
    val_set = PongSequenceDataset(val_df, args.feature_cols, seq_len=args.seq_len, allow_empty=True)
    test_set = PongSequenceDataset(test_df, args.feature_cols, seq_len=args.seq_len, allow_empty=True)

    total_sequences = len(train_set) + len(val_set) + len(test_set)
    print(
        f"[INFO] Dataset totale: {total_sequences} sequenze da {len(csv_paths)} sessioni "
        f"(train {len(train_set)}, val {len(val_set)}, test {len(test_set)})."
    )

    pin_memory = torch.cuda.is_available()
    train_loader, val_loader, test_loader = make_dataloaders(
        train_set,
        val_set,
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(df["action_idx"].unique())
    model = PongTransformer(
        input_dim=len(args.feature_cols),
        seq_len=args.seq_len,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "pong_transformer_best.pt"

    # Directory per log/predizioni
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = maybe_resume(checkpoint_path, model, optimizer, device)

    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip=args.grad_clip
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # --- LOG JSONL ---
        log_rec = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        with open(logs_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_rec) + "\n")

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f} | "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f}"
        )

        if val_loss + args.min_delta < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                val_loss,
                metadata={
                    "feature_cols": args.feature_cols,
                    "seq_len": args.seq_len,
                    "num_classes": num_classes,
                    "normalization_stats": stats,
                },
            )
            print(f"[INFO] Nuovo best model salvato in {checkpoint_path} (val_loss {val_loss:.4f}).")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("[INFO] Early stopping triggered.")
                break

    # Carica best checkpoint per test
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[INFO] Caricato il best checkpoint da {checkpoint_path}.")

    # Test finale + salvataggio predizioni per notebook
    test_size = len(test_loader.dataset)
    if test_size > 0:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"[TEST] loss: {test_loss:.4f}, acc: {test_acc:.3f} ({test_size} sequenze)")

        # --- Predizioni di test per confusion matrix / per-class accuracy ---
        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        y_prob_all: List[List[float]] = []

        softmax = nn.Softmax(dim=1)
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = softmax(logits).cpu().numpy()                  # (B, num_classes)
                preds = logits.argmax(dim=1).cpu().numpy().tolist()

                y_pred_all.extend(preds)
                y_prob_all.extend(probs.tolist())
                y_true_all.extend(batch_y.numpy().tolist())

        # Mappa classi (indice -> etichetta originale)
        unique_labels = sorted(df["action_next"].astype(int).unique())
        idx_to_label = {int(i): int(lbl) for i, lbl in enumerate(unique_labels)}

        # Salva file JSON per il notebook
        with open(logs_dir / "test_predictions.json", "w", encoding="utf-8") as f:
            json.dump(
                {"y_true": y_true_all, "y_pred": y_pred_all, "y_prob": y_prob_all},
                f,
            )
        with open(logs_dir / "label_map.json", "w", encoding="utf-8") as f:
            json.dump({"index_to_label": idx_to_label}, f)

        # Anche un riassunto metriche (facoltativo ma utile)
        with open(logs_dir / "metrics_final.json", "w", encoding="utf-8") as f:
            json.dump({"test_loss": float(test_loss), "test_acc": float(test_acc)}, f)

        print(f"[INFO] Predizioni test salvate in {logs_dir/'test_predictions.json'}")
        print(f"[INFO] Label map salvata in {logs_dir/'label_map.json'}")
    else:
        print("[WARN] Nessun set di test; riduci --test-ratio o raccogli più dati.")


if __name__ == "__main__":
    main()
