import pandas as pd
import numpy as np
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python normalize_data.py <folder> [csv_name]")
    sys.exit(1)

# 1. Ottieni la cartella da riga di comando
folder = sys.argv[1]
csv_name = sys.argv[2] if len(sys.argv) > 2 else "pong_log.csv"

# 2. Costruisci il path del CSV di input
input_csv = os.path.join(folder, csv_name)

# 3. Carica il dataset
df = pd.read_csv(input_csv)

# 4. Seleziona le colonne da normalizzare (tutte tranne 'partita')
# Puoi personalizzare questa lista se vuoi normalizzare anche le feature!
cols_to_normalize = [c for c in df.columns if c not in ['partita'] and df[c].dtype != 'O']

eps = 1e-8  # Per evitare divisione per zero

def normalize_group(gr):
    arr = gr[cols_to_normalize].values.astype(np.float32)
    # Standardizzazione
    arr_std = (arr - arr.mean(axis=0)) / (arr.std(axis=0) + eps)
    # Min-Max
    arr_minmax = (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0) + eps)
    # DataFrame
    df_std = pd.DataFrame(arr_std, columns=[f"{col}_std" for col in cols_to_normalize], index=gr.index)
    df_minmax = pd.DataFrame(arr_minmax, columns=[f"{col}_minmax" for col in cols_to_normalize], index=gr.index)
    return pd.concat([gr, df_std, df_minmax], axis=1)

# 5. Normalizza per partita se esiste la colonna 'partita'
if 'partita' in df.columns:
    df_norm = df.groupby('partita', group_keys=False).apply(normalize_group)
else:
    df_norm = normalize_group(df)

# 6. Crea cartella "normalized_data" e salva CSV lì
normalized_dir = os.path.join(folder, "normalized_data")
os.makedirs(normalized_dir, exist_ok=True)
output_csv = os.path.join(normalized_dir, f"normalized_{csv_name}")

df_norm.to_csv(output_csv, index=False)
print(f"File salvato: {output_csv}")
