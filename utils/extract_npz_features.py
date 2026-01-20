#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estrae feature da tutti i file PongNoFrameskip-v4_{i}.npz in una cartella
e salva in un CSV unico (o uno per file).

Esempi:
  # CSV unico
  python extract_pong_folder.py ./dati out_all.csv

  # CSV unico con conversione in pixel (usa width/height se presenti)
  python extract_pong_folder.py ./dati out_all.csv --pixels

  # CSV per ogni file, salvati in ./exports
  python extract_pong_folder.py ./dati ./exports --per-file

  # Se i dati sono in un array 2D 'states' con colonne note
  python extract_pong_folder.py ./dati out_all.csv --array-key states --colnames t,ball_x,ball_y,p1_y,p2_y
"""

import argparse
import os
import re
import glob
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

COMMON_ARRAY_KEYS = ["states", "traj", "data"]
COMMON_FEATURES = ["t", "ball_x", "ball_y", "p1_y", "p2_y"]
META_WIDTH_KEYS = ["width", "W"]
META_HEIGHT_KEYS = ["height", "H"]

EP_PATTERN = re.compile(r"PongNoFrameskip-v4_(\d+)\.npz$", re.IGNORECASE)

def load_npz(path: str):
    return np.load(path, allow_pickle=False)

def pick_first_key(keys: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in keys:
            return c
    return None

def extract_from_separate_keys(data) -> Optional[pd.DataFrame]:
    keys = set(data.files)
    cols = {k: np.asarray(data[k]) for k in COMMON_FEATURES if k in keys}
    if not cols:
        return None
    lengths = {k: len(v) for k, v in cols.items() if hasattr(v, "__len__")}
    if lengths and len(set(lengths.values())) > 1:
        min_len = min(lengths.values())
        cols = {k: v[:min_len] for k, v in cols.items()}
    return pd.DataFrame(cols)

def extract_from_structured_array(arr) -> Optional[pd.DataFrame]:
    if getattr(arr, "dtype", None) is not None and arr.dtype.names:
        names = arr.dtype.names
        usable = [n for n in COMMON_FEATURES if n in names]
        if usable:
            return pd.DataFrame({n: arr[n] for n in usable})
    return None

def extract_from_2d_array(arr) -> Optional[pd.DataFrame]:
    if hasattr(arr, "ndim") and arr.ndim == 2 and arr.shape[1] >= 4:
        if arr.shape[1] >= 5:
            names = ["t", "ball_x", "ball_y", "p1_y", "p2_y"]
            return pd.DataFrame(arr[:, :5], columns=names)
        else:
            names = ["ball_x", "ball_y", "p1_y", "p2_y"]
            return pd.DataFrame(arr[:, :4], columns=names)
    return None

def force_from_array_with_names(arr, colnames: List[str]) -> pd.DataFrame:
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < len(colnames):
        raise ValueError(f"Array 2D con shape {arr.shape} ma servono >= {len(colnames)} colonne.")
    return pd.DataFrame(arr[:, :len(colnames)], columns=colnames)

def coerce_pixels(df: pd.DataFrame, data, width_key: Optional[str], height_key: Optional[str]) -> pd.DataFrame:
    keys = set(data.files)
    w_key = width_key or pick_first_key(list(keys), META_WIDTH_KEYS)
    h_key = height_key or pick_first_key(list(keys), META_HEIGHT_KEYS)
    if (w_key in keys) and (h_key in keys):
        W = float(np.asarray(data[w_key]).squeeze())
        H = float(np.asarray(data[h_key]).squeeze())
        for c in ["ball_x", "ball_y", "p1_y", "p2_y"]:
            if c in df.columns:
                maxv = float(np.nanmax(df[c]))
                minv = float(np.nanmin(df[c]))
                if minv >= -0.05 and maxv <= 1.2:  # euristica normalizzazione [0,1]
                    scale = W if c.endswith("_x") else H
                    df[c + "_px"] = (df[c] * scale).round().astype("Int64")
        df["width"] = int(W)
        df["height"] = int(H)
    return df

def invert_y_if_requested(df: pd.DataFrame, data, height_key: Optional[str], do_invert: bool) -> pd.DataFrame:
    if not do_invert:
        return df
    keys = set(data.files)
    h_key = height_key or pick_first_key(list(keys), META_HEIGHT_KEYS)
    if h_key in keys:
        H = float(np.asarray(data[h_key]).squeeze())
        for c in ["ball_y", "p1_y", "p2_y"]:
            if c in df.columns:
                if c + "_px" in df.columns:
                    df[c + "_px_inverted"] = (H - df[c + "_px"]).round().astype("Int64")
                df[c + "_inverted"] = H - df[c]
    return df

def extract_one_file(path: str,
                     array_key: Optional[str],
                     colnames: Optional[List[str]],
                     pixels: bool,
                     width_key: Optional[str],
                     height_key: Optional[str],
                     invert_y: bool) -> pd.DataFrame:
    with load_npz(path) as data:
        keys = list(data.files)
        df = extract_from_separate_keys(data)

        if df is None:
            key = array_key or pick_first_key(keys, COMMON_ARRAY_KEYS)
            if key:
                arr = data[key]
                if df is None and colnames:
                    df = force_from_array_with_names(arr, colnames)
                if df is None:
                    df = extract_from_structured_array(arr)
                if df is None:
                    df = extract_from_2d_array(arr)

        if df is None:
            raise ValueError(f"Schema non riconosciuto per {os.path.basename(path)}. Chiavi: {keys}")

        if pixels:
            df = coerce_pixels(df, data, width_key, height_key)
        if invert_y:
            df = invert_y_if_requested(df, data, height_key, True)

        # Ordina colonne
        preferred = [c for c in [
            "t","ball_x","ball_y","p1_y","p2_y",
            "ball_x_px","ball_y_px","p1_y_px","p2_y_px",
            "ball_y_inverted","p1_y_inverted","p2_y_inverted",
            "ball_y_px_inverted","p1_y_px_inverted","p2_y_px_inverted",
            "width","height"
        ] if c in df.columns]
        other = [c for c in df.columns if c not in preferred]
        df = df[preferred + other]
        return df

def find_episode_number(filename: str) -> Optional[int]:
    m = EP_PATTERN.search(filename)
    if m:
        return int(m.group(1))
    return None

def main():
    ap = argparse.ArgumentParser(description="Estrai feature Pong da una cartella di .npz")
    ap.add_argument("input_dir", help="Cartella che contiene i .npz (es. PongNoFrameskip-v4_0.npz ...)")
    ap.add_argument("output", help="Percorso CSV unico o cartella di export se --per-file")
    ap.add_argument("--per-file", action="store_true", help="Salva un CSV per ogni .npz in 'output' (che deve essere una cartella).")
    ap.add_argument("--pattern", default="PongNoFrameskip-v4_*.npz", help="Pattern glob dei file da processare.")
    ap.add_argument("--start", type=int, default=None, help="Indice episodio minimo (incluso) da includere.")
    ap.add_argument("--end", type=int, default=None, help="Indice episodio massimo (incluso) da includere.")
    ap.add_argument("--array-key", help="Chiave dell'array principale (se combinato).")
    ap.add_argument("--colnames", help="Nomi colonne per l'array 2D, comma-separati.")
    ap.add_argument("--pixels", action="store_true", help="Crea colonne *_px usando width/height se presenti.")
    ap.add_argument("--width-key", help="Nome chiave per width (se diverso).")
    ap.add_argument("--height-key", help="Nome chiave per height (se diverso).")
    ap.add_argument("--invert-y", action="store_true", help="Inverte Y (origine in alto). Richiede height.")
    args = ap.parse_args()

    pattern_path = os.path.join(args.input_dir, args.pattern)
    files = sorted(glob.glob(pattern_path))

    # Filtra per range episodio se richiesto
    selected = []
    for f in files:
        ep = find_episode_number(os.path.basename(f))
        if ep is None:
            continue
        if args.start is not None and ep < args.start:
            continue
        if args.end is not None and ep > args.end:
            continue
        selected.append((ep, f))

    if not selected:
        raise SystemExit(f"Nessun file trovato con pattern {pattern_path} nel range richiesto.")

    # Ordina per episodio
    selected.sort(key=lambda x: x[0])
    print(f"Trovati {len(selected)} file. Esempio: {os.path.basename(selected[0][1])} … {os.path.basename(selected[-1][1])}")

    colnames = [c.strip() for c in args.colnames.split(",")] if args.colnames else None

    if args.per_file:
        # Output per-file
        os.makedirs(args.output, exist_ok=True)
        ok, fail = 0, 0
        for ep, path in selected:
            try:
                df = extract_one_file(path, args.array_key, colnames, args.pixels, args.width_key, args.height_key, args.invert_y)
                df.insert(0, "episode", ep)
                df.insert(1, "source_file", os.path.basename(path))
                out_path = os.path.join(args.output, f"PongNoFrameskip-v4_{ep}.csv")
                df.to_csv(out_path, index=False)
                ok += 1
                print(f"[{ep}] {os.path.basename(path)} -> {out_path} ({len(df)} righe)")
            except Exception as e:
                fail += 1
                print(f"[{ep}] {os.path.basename(path)}: {e}")
        print(f"Completato. Successi: {ok}, Falliti: {fail}")
    else:
        # CSV unico concatenato
        frames = []
        ok, fail = 0, 0
        for ep, path in selected:
            try:
                df = extract_one_file(path, args.array_key, colnames, args.pixels, args.width_key, args.height_key, args.invert_y)
                df.insert(0, "episode", ep)
                df.insert(1, "source_file", os.path.basename(path))
                frames.append(df)
                ok += 1
                print(f"[{ep}] {os.path.basename(path)} ({len(df)} righe)")
            except Exception as e:
                fail += 1
                print(f"[{ep}] {os.path.basename(path)}: {e}")

        if not frames:
            raise SystemExit("Tutti i file sono falliti: nessun output prodotto.")
        big = pd.concat(frames, axis=0, ignore_index=True)

        # Ordina colonne mettendo episode/source_file davanti
        cols = list(big.columns)
        front = [c for c in ["episode", "source_file"] if c in cols]
        rest = [c for c in cols if c not in front]
        big = big[front + rest]

        out_csv = args.output
        big.to_csv(out_csv, index=False)
        print(f"CSV unico salvato: {out_csv}  (righe: {len(big)}, episodi: {ok}, falliti: {fail})")

if __name__ == "__main__":
    main()
