import os
import sys
import subprocess

# Percorso assoluto della cartella src: ci serve per costruire percorsi robusti verso gli altri script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_folder(folder: str) -> str:
    """Restituisce il percorso assoluto verso la cartella dei dati."""
    return folder if os.path.isabs(folder) else os.path.join(SCRIPT_DIR, folder)


def estrai_tutto():
    # Richiamiamo lo script di estrazione con percorso assoluto, cosi' funziona da qualunque working directory.
    script_path = os.path.join(SCRIPT_DIR, "extract_all_features.py")
    result = subprocess.run(["python3", script_path])
    if result.returncode == 0:
        print("\nEstrazione COMPLETATA per tutte le partite!")
    else:
        print("\nQualcosa e' andato storto durante l'estrazione!")


def preprocess_and_normalize():
    folder = input("Inserisci il nome della cartella dati (es: game_data_20250728): ").strip()
    folder_path = _resolve_folder(folder)
    csv_in = os.path.join(folder_path, "pong_data_features.csv")
    if not os.path.exists(csv_in):
        print(f"File non trovato: {csv_in}")
        return

    preprocessing_script = os.path.join(SCRIPT_DIR, "preprocessing.py")
    result1 = subprocess.run(["python3", preprocessing_script, folder_path])
    if result1.returncode != 0:
        print("Errore nel preprocessing!")
        return

    preproc_path = os.path.join(folder_path, "pong_data_features_preprocessed.csv")
    if not os.path.exists(preproc_path):
        print(f"File preprocessato non trovato: {preproc_path}")
        return

    normalize_script = os.path.join(SCRIPT_DIR, "normalize_data.py")
    result2 = subprocess.run(["python3", normalize_script, folder_path])
    if result2.returncode == 0:
        print("Preprocessing e normalizzazione COMPLETATI!")
        print(f"Dati finali in: {os.path.join(folder_path, 'pong_data_features_preprocessed_normalized.csv')}")
    else:
        print("Errore nella normalizzazione!")


def test_dataset():
    folder = input("Inserisci il nome della cartella dati (es: game_data_20250728): ").strip()
    folder_path = _resolve_folder(folder)
    test_script = os.path.join(SCRIPT_DIR, "test_dataset.py")
    result = subprocess.run(["python3", test_script, folder_path])
    if result.returncode != 0:
        print("Dataset corrotto")


def allena_modello():
    folder = input("Inserisci il nome della cartella dati (es: game_data_20250728): ").strip()
    folder_path = _resolve_folder(folder)
    training_script = os.path.join(SCRIPT_DIR, "NN", "training.py")
    result = subprocess.run(["python3", training_script, folder_path])
    if result.returncode == 0:
        print("Training completato!")
    else:
        print("Errore durante il training!")


def valuta_modello():
    folder = input("Inserisci la cartella del nuovo database (es: game_data_20250801): ").strip()
    folder_path = _resolve_folder(folder)
    nomefile = "pong_data_features_preprocessed_normalized.csv"
    file_path = os.path.join(folder_path, nomefile)
    if not os.path.exists(file_path):
        print(f"File non trovato: {file_path}")
        return

    # Individuiamo lo script di valutazione usando percorsi assoluti.
    test_model_paths = [
        os.path.join(SCRIPT_DIR, "NN", "test_model.py"),
        os.path.join(SCRIPT_DIR, "test_model.py")
    ]
    for path in test_model_paths:
        if os.path.exists(path):
            test_model_path = path
            break
    else:
        print("Non trovo il file test_model.py! Controlla dove si trova.")
        return

    result = subprocess.run(["python3", test_model_path, file_path])
    if result.returncode == 0:
        print("Valutazione completata!")
    else:
        print("Errore durante la valutazione!")


def esci():
    print("Ciao!")
    sys.exit()


if __name__ == "__main__":
    azioni = {
        "1": ("Estrai TUTTO (file unico: pallina + paddle + opponent, automatico)", estrai_tutto),
        "2": ("Preprocessing + Normalizza", preprocess_and_normalize),
        "3": ("Test automatici sul dataset", test_dataset),
        "4": ("Allena modello", allena_modello),
        "5": ("Valuta modello su un nuovo database", valuta_modello),
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
