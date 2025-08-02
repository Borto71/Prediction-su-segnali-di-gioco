# main.py (versione smart, solo percorso cartella richiesto)

import os
import sys
import subprocess

def estrai_tutto():
    result = subprocess.run(["python3", "extract_all_features.py"])
    if result.returncode == 0:
        print("\nEstrazione COMPLETATA per tutte le partite!")
    else:
        print("\nQualcosa è andato storto durante l'estrazione!")

def normalizza():
    folder = input("Inserisci il nome della cartella dati (es: game_data_20250728): ").strip()
    nomefile = "pong_data_features_preprocessed.csv"
    file_path = os.path.join(folder, nomefile)
    if not os.path.exists(file_path):
        print(f"File non trovato: {file_path}")
        return
    # Nuova chiamata: passa cartella e nomefile
    result = subprocess.run(["python3", "normalize_data.py", folder, nomefile])
    if result.returncode == 0:
        print("Normalizzazione COMPLETATA!")
        print(f"Dati normalizzati in: {os.path.join(folder, nomefile.replace('.csv', '_normalized.csv'))}")
    else:
        print("Errore nella normalizzazione!")

def preprocess_and_normalize():
    folder = input("Inserisci il nome della cartella dati (es: game_data_20250728): ").strip()
    csv_in = os.path.join(folder, "pong_data_features.csv")
    if not os.path.exists(csv_in):
        print(f"File non trovato: {csv_in}")
        return
    # Preprocessing: automatico, output file fisso
    result1 = subprocess.run(["python3", "preprocessing.py", folder])
    if result1.returncode != 0:
        print("Errore nel preprocessing!")
        return
    preproc_file = "pong_data_features_preprocessed.csv"
    preproc_path = os.path.join(folder, preproc_file)
    if not os.path.exists(preproc_path):
        print(f"File preprocessato non trovato: {preproc_path}")
        return
    # Normalizzazione: automatico, output file fisso
    result2 = subprocess.run(["python3", "normalize_data.py", folder, preproc_file])
    if result2.returncode == 0:
        print("Preprocessing e normalizzazione COMPLETATI!")
        print(f"Dati finali in: {os.path.join(folder, preproc_file.replace('.csv', '_normalized.csv'))}")
    else:
        print("Errore nella normalizzazione!")

def allena_modello():
    os.system("python3 NN/training.py")

def esci():
    print("Ciao!")
    sys.exit()

if __name__ == "__main__":
    azioni = {
        "1": ("Estrai TUTTO (file unico: pallina + paddle + opponent, automatico)", estrai_tutto),
        "2": ("Normalizza dati", normalizza),
        "3": ("Preprocessing + Normalizza", preprocess_and_normalize),
        "4": ("Allena modello", allena_modello),
        "0": ("Esci", esci)
    }

    while True:
        print("\n--- MAIN PREDICTION PONG ---")
        for k, (desc, _) in azioni.items():
            print(f"{k}: {desc}")
        scelta = input("Cosa vuoi fare? ")
        if scelta in azioni:
            azioni[scelta][1]()
        else:
            print("Scelta non valida!")
