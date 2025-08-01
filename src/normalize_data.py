import pandas as pd
import os
import sys

if len(sys.argv) >= 2:
    folder = sys.argv[1]
else:
    folder = input("Inserisci la cartella dati (es: game_data_20250801): ").strip()

if len(sys.argv) >= 3:
    input_csv = sys.argv[2]
else:
    input_csv = input("Nome file da normalizzare (es: pong_data_features_preprocessed.csv): ").strip()

input_csv_path = os.path.join(folder, input_csv)
if not os.path.exists(input_csv_path):
    print(f"File non trovato: {input_csv_path}")
    sys.exit(1)

df = pd.read_csv(input_csv_path)

# Scegli quali colonne NON normalizzare (es: label, azioni, partita, reward)
cols_to_exclude = ["partita", "action", "action_next", "reward", "score_left", "score_right"]
cols_to_normalize = [col for col in df.columns if col not in cols_to_exclude and df[col].dtype != "O"]

print("Colonne da normalizzare:", cols_to_normalize)

# Normalizzazione standard (z-score: media 0, dev std 1)
for col in cols_to_normalize:
    mean = df[col].mean()
    std = df[col].std()
    if std == 0:
        print(f"Attenzione: std nulla per {col}, salto normalizzazione")
        continue
    df[col + "_std"] = (df[col] - mean) / std

# Normalizzazione min-max (0-1)
for col in cols_to_normalize:
    minv = df[col].min()
    maxv = df[col].max()
    if maxv - minv == 0:
        print(f"Attenzione: min=max per {col}, salto minmax")
        continue
    df[col + "_minmax"] = (df[col] - minv) / (maxv - minv)

# Crea output dir se non esiste
out_dir = os.path.join(folder, "normalized_data")
os.makedirs(out_dir, exist_ok=True)
basename = os.path.basename(input_csv)
out_csv = os.path.join(out_dir, "normalized_" + basename)
df.to_csv(out_csv, index=False)
print(f"File normalizzato salvato in: {out_csv}")

# Controllo range output
print("\nRange colonne normalizzate (std):")
for col in [c for c in df.columns if c.endswith("_std")]:
    print(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}, std={df[col].std():.2f}")
print("\nRange colonne normalizzate (minmax):")
for col in [c for c in df.columns if c.endswith("_minmax")]:
    print(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}")
