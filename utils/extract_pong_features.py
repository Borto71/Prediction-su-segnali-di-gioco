#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import csv
import numpy as np
from tqdm import tqdm

# Colori target in RGB (come da tua nota)
BALL_COLOR = np.array([236, 236, 236], dtype=np.uint8)       # bianco palla
RIGHT_PADDLE_COLOR = np.array([92, 186, 92], dtype=np.uint8) # verde racchetta destra
LEFT_PADDLE_COLOR  = np.array([213, 130, 74], dtype=np.uint8) # arancione racchetta sinistra

def to_hwc_uint8(frame_raw):
    """
    Converte un frame in array uint8 HxWx3 quando possibile.
    Gestisce casi comuni:
      - HxWx3 RGB float [0..1] / [0..255] -> uint8
      - HxWx4 -> usa i primi 3 canali
      - 3xHxW (CHW) -> trasponi a HxWx3
      - HxW (grayscale) -> duplica a 3 canali
      - HxWx4 stack di 4 grigi -> usa max sui canali e duplica a 3 canali
      - dtype=object -> np.asarray
    Se non è interpretabile, ritorna None.
    """
    if frame_raw is None:
        return None
    arr = np.asarray(frame_raw)
    if arr.size == 0:
        return None
    arr = np.squeeze(arr)

    # 2D: grigio -> 3 canali
    if arr.ndim == 2:
        gray = arr
        gray = _to_uint8(gray)
        img = np.stack([gray, gray, gray], axis=-1)
        return img

    # 3D
    if arr.ndim == 3:
        # HWC?
        if arr.shape[-1] in (1, 3, 4) and arr.shape[0] >= 8 and arr.shape[1] >= 8:
            if arr.shape[-1] == 1:
                gray = _to_uint8(arr[..., 0])
                return np.stack([gray, gray, gray], axis=-1)
            elif arr.shape[-1] == 3:
                return _to_uint8(arr)
            elif arr.shape[-1] == 4:
                return _to_uint8(arr[..., :3])
        # CHW?
        if arr.shape[0] in (1, 3, 4) and arr.shape[1] >= 8 and arr.shape[2] >= 8:
            if arr.shape[0] == 1:
                gray = _to_uint8(arr[0])
                return np.stack([gray, gray, gray], axis=-1)
            elif arr.shape[0] >= 3:
                rgb = arr[:3].transpose(1, 2, 0)  # (3,H,W) -> (H,W,3)
                return _to_uint8(rgb)

        # caso Atari comune: (H,W,4) stack di grigi -> max e poi duplica
        if arr.shape[-1] == 4 and arr.shape[0] >= 8 and arr.shape[1] >= 8:
            gray = np.max(arr, axis=-1)
            gray = _to_uint8(gray)
            return np.stack([gray, gray, gray], axis=-1)

    return None

def _to_uint8(x):
    """Converte array in uint8 gestendo float [0..1]/[0..255] e int."""
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        if xmax <= 1.0 + 1e-6:   # [0..1]
            x = x * 255.0
        elif xmax > 255.0 or xmin < 0.0:
            # normalizza robustamente
            rng = (xmax - xmin)
            if rng > 1e-9:
                x = (x - xmin) / rng * 255.0
            else:
                x = np.zeros_like(x)
    x = np.clip(x, 0, 255).astype(np.uint8, copy=False)
    return x

def color_mask(img_hwc_uint8, color_rgb_uint8, tol=30):
    """
    Ritorna maschera booleana dei pixel 'vicini' a color_rgb_uint8 con tolleranza Euclidea in RGB.
    img: HxWx3 uint8
    """
    diff = img_hwc_uint8.astype(np.int16) - color_rgb_uint8.reshape(1, 1, 3).astype(np.int16)
    dist2 = (diff * diff).sum(axis=2)  # distanza euclidea^2
    return dist2 <= (tol * tol)

def extract_from_color(img_hwc_uint8, tol_ball=20, tol_paddle=28):
    """
    Rileva (ball_x, ball_y, paddle1_y, paddle2_y) via color-matching.
    - palla: bianco ~ [236,236,236]
    - paddle sx: arancione ~ [213,130,74]
    - paddle dx: verde ~ [92,186,92]
    Ritorna NaN se non rilevato.
    """
    H, W, _ = img_hwc_uint8.shape

    # Maschere colore
    m_ball  = color_mask(img_hwc_uint8, BALL_COLOR, tol=tol_ball)
    m_left  = color_mask(img_hwc_uint8, LEFT_PADDLE_COLOR, tol=tol_paddle)
    m_right = color_mask(img_hwc_uint8, RIGHT_PADDLE_COLOR, tol=tol_paddle)

    # Palla: ignora margini per non confonderla con paddle
    ys_b, xs_b = np.where(m_ball)
    ball_x = ball_y = np.nan
    if xs_b.size > 0:
        inner = (xs_b >= 8) & (xs_b <= W - 9)
        if np.any(inner):
            ball_x = float(np.median(xs_b[inner]))
            ball_y = float(np.median(ys_b[inner]))
        else:
            # fallback: prendi tutto (può capitare vicino ai bordi)
            ball_x = float(np.median(xs_b))
            ball_y = float(np.median(ys_b))

    # Paddle sinistro: pixel arancioni vicino al bordo sinistro
    ys_l, xs_l = np.where(m_left)
    paddle1_y = np.nan
    if xs_l.size > 0:
        near_left = xs_l < 16  # bordo sinistro
        if np.any(near_left):
            paddle1_y = float(np.median(ys_l[near_left]))
        else:
            paddle1_y = float(np.median(ys_l))

    # Paddle destro: pixel verdi vicino al bordo destro
    ys_r, xs_r = np.where(m_right)
    paddle2_y = np.nan
    if xs_r.size > 0:
        near_right = xs_r > (W - 17)  # bordo destro
        if np.any(near_right):
            paddle2_y = float(np.median(ys_r[near_right]))
        else:
            paddle2_y = float(np.median(ys_r))

    return ball_x, ball_y, paddle1_y, paddle2_y

def process_npz_file(npz_path, writer, tol_ball, tol_paddle, pattern_key_obs="obs"):
    """
    Legge un .npz e scrive una riga per frame:
    file, frame, ball_x, ball_y, paddle1_y, paddle2_y, action, reward, episode_start
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    if pattern_key_obs not in data:
        raise ValueError(f"{os.path.basename(npz_path)}: chiave '{pattern_key_obs}' mancante. Presenti: {keys}")

    obs = data[pattern_key_obs]
    actions = data["taken actions"] if "taken actions" in data else data.get("actions")
    if actions is None:
        raise ValueError(f"{os.path.basename(npz_path)}: chiave 'taken actions' mancante. Presenti: {keys}")

    rewards = data["rewards"] if "rewards" in data else np.zeros(len(actions), dtype=float)
    starts  = data["episode_starts"] if "episode_starts" in data else np.zeros(len(actions), dtype=bool)

    n = min(len(obs), len(actions), len(rewards), len(starts))
    fname = os.path.basename(npz_path)

    for i in range(n):
        frame_raw = obs[i]
        img = to_hwc_uint8(frame_raw)
        if img is None or img.ndim != 3 or img.shape[2] != 3:
            # non interpretabile in colore -> NaN
            ball_x = ball_y = paddle1_y = paddle2_y = np.nan
        else:
            ball_x, ball_y, paddle1_y, paddle2_y = extract_from_color(
                img, tol_ball=tol_ball, tol_paddle=tol_paddle
            )

        writer.writerow([
            fname,
            i,
            ball_x,
            ball_y,
            paddle1_y,
            paddle2_y,
            int(actions[i]),
            float(rewards[i]),
            int(bool(starts[i])),
        ])

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Estrai (palla, paddle, azione, reward) da Pong .npz via color-matching.")
    parser.add_argument("input_dir", help="Cartella con i file .npz (es. PongNoFrameskip-v4_*.npz)")
    parser.add_argument("output_csv", help="Percorso CSV di output")
    parser.add_argument("--pattern", default="PongNoFrameskip-v4_*.npz", help="Glob pattern per i file .npz")
    parser.add_argument("--obs-key", default="obs", help="Chiave dell'array di osservazioni nel .npz (default: 'obs')")
    parser.add_argument("--tol-ball", type=int, default=20, help="Tolleranza colore per palla (RGB, default 20)")
    parser.add_argument("--tol-paddle", type=int, default=28, help="Tolleranza colore per paddle (RGB, default 28)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        print(f"Nessun file trovato in {args.input_dir} con pattern {args.pattern}")
        return

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file",
            "frame",
            "ball_x",
            "ball_y",
            "paddle1_y",
            "paddle2_y",
            "action",
            "reward",
            "episode_start"
        ])
        for npz_path in tqdm(files, desc="Estrazione Pong features (color)"):
            try:
                process_npz_file(npz_path, writer, args.tol_ball, args.tol_paddle, pattern_key_obs=args.obs_key)
            except Exception as e:
                print(f"[WARN] {os.path.basename(npz_path)}: {e}")

    print(f"\n Estrazione completata! CSV: {args.output_csv}")

if __name__ == "__main__":
    main()
