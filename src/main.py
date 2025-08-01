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
    folder = input("Inserisci il nome della cartella dati (es: game_data_20250728): ")
    nomefile = input("Nome file da normalizzare (default: pong_data_features_preprocessed.csv): ").strip()
    if nomefile == "":
        nomefile = "pong_data_features_preprocessed.csv"
    file_path = os.path.join(folder, nomefile)
    if not os.path.exists(file_path):
        print(f"File non trovato: {file_path}")
        return
    result = subprocess.run(["python3", "normalize_data.py", folder, nomefile])
    if result.returncode == 0:
        print("Normalizzazione COMPLETATA!")
    else:
        print("Errore nella normalizzazione!")

def preprocess_and_normalize():
    folder = input("Inserisci il nome della cartella dati (es: game_data_20250728): ")
    nomefile = input("Nome file di partenza (default: pong_data_features.csv): ").strip()
    if nomefile == "":
        nomefile = "pong_data_features.csv"
    input_features = os.path.join(folder, nomefile)
    if not os.path.exists(input_features):
        print(f"File non trovato: {input_features}")
        return
    result1 = subprocess.run(["python3", "preprocessing.py", input_features])
    if result1.returncode != 0:
        print("Errore nel preprocessing!")
        return
    preproc_file = nomefile.replace(".csv", "_preprocessed.csv")
    preproc_path = os.path.join(folder, preproc_file)
    if not os.path.exists(preproc_path):
        print(f"File preprocessato non trovato: {preproc_path}")
        return
    result2 = subprocess.run(["python3", "normalize_data.py", folder, preproc_file])
    if result2.returncode == 0:
        print("Preprocessing e normalizzazione COMPLETATI!")
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
