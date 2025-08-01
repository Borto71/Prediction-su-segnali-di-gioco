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
    subprocess.run(["python3", "normalize_data.py", folder])

def preprocess_and_normalize():
    folder = input("Inserisci il nome della cartella dati (es: game_data_20250728): ")
    input_features = os.path.join(folder, "pong_data_features.csv")
    result1 = subprocess.run(["python3", "preprocessing.py", input_features])
    if result1.returncode != 0:
        print("Errore nel preprocessing!")
        return
    preproc_file = input_features.replace(".csv", "_preprocessed.csv")
    result2 = subprocess.run(["python3", "normalize_data.py", folder, os.path.basename(preproc_file)])
    if result2.returncode == 0:
        print("Preprocessing e normalizzazione COMPLETATI!")
    else:
        print("Errore nella normalizzazione!")

def allena_modello():
    # Stampa in tempo reale durante l'allenamento!
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
