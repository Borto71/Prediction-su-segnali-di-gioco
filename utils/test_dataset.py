# Test automatici sul dataset per verificare la qualità dei dati
import pandas as pd
import sys
import os

# Controlla che venga passato almeno un argomento da linea di comando
if len(sys.argv) < 2:
    raise ValueError("Uso: python preprocessing.py <nome_cartella>")

CSV_PATH = os.path.join(sys.argv[1].strip(), "pong_data_features_preprocessed_normalized.csv")

df = pd.read_csv(CSV_PATH)

# TEST 1: Nessun NaN
print("[Test 1] Verifica assenza di NaN...")
assert not df.isnull().values.any(), "Errore: ci sono valori NaN nel dataset!"
print("Nessun NaN trovato.")

# TEST 2: Colonne richieste presenti
print("[Test 2] Verifica colonne obbligatorie...")
required = ['ball_x', 'ball_y', 'right_paddle_y', 'action_next']
missing = [col for col in required if col not in df.columns]
assert not missing, f"Mancano le seguenti colonne: {missing}"
print("Tutte le colonne richieste sono presenti.")

# TEST 3: Tutti i valori numerici sono tra 0 e 1
print("[Test 3] Verifica che i valori siano tra 0 e 1 (feature normalizzate)...")

# Prendi solo le colonne che terminano con '_minmax'
minmax_cols = [col for col in df.columns if col.endswith('_minmax')]

out_of_bounds = [
    col for col in minmax_cols if not df[col].between(0, 1).all()
]

assert not out_of_bounds, f"Le seguenti colonne hanno valori fuori dal range [0,1]: {out_of_bounds}"
print("Tutti i valori normalizzati (_minmax) sono nel range [0,1].")

# TEST 4: Tutte le classi presenti
print("[Test 4] Verifica distribuzione classi in 'action_next'...")
counts = df['action_next'].value_counts()
assert counts.min() > 0, f"Una o più classi sono assenti in 'action_next': {counts.to_dict()}"
print(f"Tutte le classi sono presenti: {counts.to_dict()}")

print("\nTutti i test superati con successo!")