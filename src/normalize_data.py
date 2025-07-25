import pandas as pd
import numpy as np
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python normalize_data.py <folder>")
    sys.exit(1)

# 1. Ottieni la cartella da riga di comando
folder = sys.argv[1]

# 2. Costruisci il path del CSV di input
input_csv = os.path.join(folder, "pong_data_features.csv")

# 3. Carica il dataset
df = pd.read_csv(input_csv)

# 4. Seleziona le colonne da normalizzare
features = ['ball_x', 'ball_y', 'right_paddle_y', 'left_paddle_y']
X = df[features].values.astype(np.float32)

# 5. Standardizzazione (media 0, std 1)
X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)

# 6. Min-Max Scaling (0-1)
X_minmax = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# 7. Crea nuovi DataFrame con nomi colonne aggiornati
df_standardized = pd.DataFrame(X_standardized, columns=[f"{col}_std" for col in features])
df_minmax = pd.DataFrame(X_minmax, columns=[f"{col}_minmax" for col in features])

# 8. Unisci tutto in un unico DataFrame
df_normalized = pd.concat([df, df_standardized, df_minmax], axis=1)

# 9. Salva in CSV nella stessa cartella
output_csv = os.path.join(folder, "normalized_data3.csv")
df_normalized.to_csv(output_csv, index=False)

print(f"File salvato: {output_csv}")
