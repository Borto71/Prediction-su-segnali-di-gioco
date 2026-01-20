#!/usr/bin/env python3
"""
Estrae le feature (posizioni palla/racchette) per tutte le partite di una cartella `game_data_*`.

Uso:
    python utils/extract_all_features.py data/game_data_20251014
    python utils/extract_all_features.py 20251014               # equivalente (aggiunge il prefisso game_data_)
    python utils/extract_all_features.py game_data_20251014     # funziona sia relativo che assoluto
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

# --- Costanti colore / coordinate (coerenti con gli altri script) ----------
BALL_COLOR = np.array([236, 236, 236])        # Bianco della palla Pong
RIGHT_PADDLE_COLOR = np.array([92, 186, 92])  # Verde racchetta destra
LEFT_PADDLE_COLOR = np.array([213, 130, 74])  # Arancione racchetta sinistra

Y_MIN, Y_MAX = 34, 194
X_MIN_BALL, X_MAX_BALL = 10, 150
X_MIN_RIGHT, X_MAX_RIGHT = 127, 158
X_MIN_LEFT, X_MAX_LEFT = 10, 25


# --- CLI -------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estrai feature dai replay salvati in una cartella game_data.")
    parser.add_argument(
        "target",
        type=str,
        help="Percorso (relativo o assoluto) o suffisso data della cartella game_data.",
    )
    return parser.parse_args()


def resolve_game_dir(target: str) -> Path:
    candidate = Path(target)
    if candidate.exists():
        return candidate.resolve()

    # Se viene passato solo il suffisso data (es. 20251014)
    suffix_dir = Path(f"game_data_{target}")
    if suffix_dir.exists():
        return suffix_dir.resolve()

    raise FileNotFoundError(f"Cartella game_data non trovata per l'input: {target}")


def resolve_paths(base_dir: Path) -> Tuple[Path, Path, Path]:
    replay_dir = base_dir / "replay"
    log_path = base_dir / "pong_log.csv"
    features_path = base_dir / "pong_data_features.csv"
    return replay_dir, log_path, features_path


# --- Estrazione singolo frame ----------------------------------------------
def extract_ball(frame: np.ndarray) -> Tuple[int, int]:
    cropped = frame[Y_MIN:Y_MAX, X_MIN_BALL:X_MAX_BALL, :]
    mask = np.all(cropped == BALL_COLOR, axis=2)
    ys, xs = np.where(mask)
    if xs.size > 0:
        ball_x = int(xs.mean()) + X_MIN_BALL
        ball_y = int(ys.mean()) + Y_MIN
    else:
        ball_x = ball_y = -1
    return ball_x, ball_y


def extract_right_paddle(frame: np.ndarray) -> int:
    cropped = frame[Y_MIN:Y_MAX, X_MIN_RIGHT:X_MAX_RIGHT, :]
    mask = np.all(cropped == RIGHT_PADDLE_COLOR, axis=2)
    ys, _ = np.where(mask)
    if ys.size > 0:
        return int(ys.mean()) + Y_MIN
    return -1


def extract_left_paddle(frame: np.ndarray) -> int:
    cropped = frame[Y_MIN:Y_MAX, X_MIN_LEFT:X_MAX_LEFT, :]
    mask = np.all(cropped == LEFT_PADDLE_COLOR, axis=2)
    ys, _ = np.where(mask)
    if ys.size > 0:
        return int(ys.mean()) + Y_MIN
    return -1


# --- Iterazione sulle partite ----------------------------------------------
def iter_partite(replay_dir: Path) -> Iterable[Tuple[int, Path]]:
    for file in sorted(replay_dir.glob("partita_*.npz")):
        parts = file.stem.split("_")
        if len(parts) < 2:
            continue
        try:
            partita_id = int(parts[1])
        except ValueError:
            continue
        yield partita_id, file


# --- Main ------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    base_dir = resolve_game_dir(args.target)
    replay_dir, log_path, features_path = resolve_paths(base_dir)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file {log_path} non trovato!")
    if not replay_dir.exists():
        raise FileNotFoundError(f"Directory replay {replay_dir} non trovata!")

    print(f"[INFO] Carico log da: {log_path}")
    df_log = pd.read_csv(log_path).reset_index(drop=True)
    if df_log.empty:
        raise ValueError(f"Log {log_path} vuoto.")

    for col in ("ball_x", "ball_y", "right_paddle_y", "left_paddle_y"):
        if col not in df_log.columns:
            df_log[col] = np.nan

    partite_log = sorted(df_log["partita"].dropna().unique())
    if not partite_log:
        raise ValueError("Nessuna partita nel log (colonna 'partita' vuota).")

    print(f"[INFO] Partite nel log: {partite_log}")

    for partita_id in partite_log:
        replay_path = replay_dir / f"partita_{int(partita_id)}.npz"
        if not replay_path.exists():
            print(f"[WARN] Replay {replay_path} assente, salto.")
            continue

        frames = np.load(replay_path)
        mask = df_log["partita"] == partita_id
        idx_log = df_log.index[mask]

        if len(idx_log) == 0:
            print(f"[WARN] Nessuna entry log per partita {partita_id}, salto.")
            continue

        if len(idx_log) != len(frames):
            print(
                f"[WARN] Partita {partita_id}: {len(frames)} frame ma {len(idx_log)} righe di log; "
                "allineo fino a min(len)."
            )

        step_min = min(len(frames), len(idx_log))
        ball_xs, ball_ys, right_ys, left_ys = [], [], [], []

        for i in range(step_min):
            frame = frames[i]
            bx, by = extract_ball(frame)
            ball_xs.append(bx)
            ball_ys.append(by)
            right_ys.append(extract_right_paddle(frame))
            left_ys.append(extract_left_paddle(frame))

        df_log.loc[idx_log[:step_min], "ball_x"] = ball_xs
        df_log.loc[idx_log[:step_min], "ball_y"] = ball_ys
        df_log.loc[idx_log[:step_min], "right_paddle_y"] = right_ys
        df_log.loc[idx_log[:step_min], "left_paddle_y"] = left_ys

    df_log = df_log.sort_values(["partita", "step"]).reset_index(drop=True)
    df_log.to_csv(features_path, index=False)
    print(f"[INFO] File feature master creato in: {features_path}")


if __name__ == "__main__":
    main()
