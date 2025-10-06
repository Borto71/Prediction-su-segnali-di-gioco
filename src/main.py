import os
import sys
import subprocess
from tkinter import *

def estrai_tutto():
    result = subprocess.run(["python3", "extract_all_features.py"])
    if result.returncode == 0:
        print("\nEstrazione COMPLETATA per tutte le partite!")
    else:
        print("\nQualcosa è andato storto durante l'estrazione!")

#def normalizza():
#    folder = input("Inserisci il nome della cartella dati (es: game_data_20250728): ").strip()
#    nomefile = "pong_data_features_preprocessed.csv"
#    file_path = os.path.join(folder, nomefile)
#    if not os.path.exists(file_path):
#        print(f"File non trovato: {file_path}")
#        return
#    result = subprocess.run(["python3", "normalize_data.py", folder, nomefile])
#    if result.returncode == 0:
#        print("Normalizzazione COMPLETATA!")
#        print(f"Dati normalizzati in: {os.path.join(folder, nomefile.replace('.csv', '_normalized.csv'))}")
#    else:
#        print("Errore nella normalizzazione!")

def preprocess_and_normalize():
    folder = input("Inserisci il nome della cartella dati (es: game_data_20250728): ").strip()
    csv_in = os.path.join(folder, "pong_data_features.csv")
    if not os.path.exists(csv_in):
        print(f"File non trovato: {csv_in}")
        return
    result1 = subprocess.run(["python3", "preprocessing.py", folder])
    if result1.returncode != 0:
        print("Errore nel preprocessing!")
        return
    preproc_path = os.path.join(folder, "pong_data_features_preprocessed.csv")
    if not os.path.exists(preproc_path):
        print(f"File preprocessato non trovato: {preproc_path}")
        return
    result2 = subprocess.run(["python3", "normalize_data.py", folder])
    if result2.returncode == 0:
        print("Preprocessing e normalizzazione COMPLETATI!")
        print(f"Dati finali in: {os.path.join(folder, 'pong_data_features_preprocessed_normalized.csv')}")
    else:
        print("Errore nella normalizzazione!")

def test_dataset():
    folder = input("Inserisci il nome della cartella dati (es: game_data_20250728): ").strip()
    result = subprocess.run(["python3", "test_dataset.py", folder])
    if result.returncode != 0:
        print("Dataset corrotto")

def allena_modello():
    folder = input("Inserisci il nome della cartella dati (es: game_data_20250728): ").strip()
    result = subprocess.run(["python3", "NN/training.py", folder])
    if result.returncode == 0:
        print("Training completato!")
    else:
        print("Errore durante il training!")

def valuta_modello():
    folder = input("Inserisci la cartella del nuovo database (es: game_data_20250801): ").strip()
    nomefile = "pong_data_features_preprocessed_normalized.csv"
    file_path = os.path.join(folder, nomefile)
    if not os.path.exists(file_path):
        print(f"File non trovato: {file_path}")
        return

    # Autodetect path per test_model.py
    test_model_paths = [
        os.path.join("src", "NN", "test_model.py"),
        os.path.join("NN", "test_model.py"),
        "test_model.py"
    ]
    found = False
    for path in test_model_paths:
        if os.path.exists(path):
            test_model_path = path
            found = True
            break
    if not found:
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
  
    # Create object 
    root = Tk() 
    
    # Adjust size 
    root.geometry("500x500")
    root.title("Prediction Pong")

    # Configure grid columns and rows to expand properly
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=2)
    root.grid_columnconfigure(2, weight=1)
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=50)
    root.grid_rowconfigure(2, weight=50)
    root.grid_rowconfigure(3, weight=50)

    # Create welcome label and adjust font
    welcomeLabel = Label(root, text="Benvenuto in Prediction Pong!", font=('Helvetica', 16, 'bold'), anchor="center", justify="center")
    welcomeLabel.grid(row=0, column=0, padx=10, pady=10, columnspan=3, sticky="nsew")

    newGameBtn = Button(root, text="Gioca a Pong", command=lambda: subprocess.run(["python3", "play_pong.py"]), height=2, width=20)
    newGameBtn.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

    nextBtn = Button(root, text="Avanti", height=2, width=50)
    nextBtn.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")

    # Execute tkinter
    root.mainloop()


    azioni = {
        "1": ("Estrai TUTTO (file unico: pallina + paddle + opponent, automatico)", estrai_tutto),
        #"2": ("Normalizza dati", normalizza),
        "2": ("Preprocessing + Normalizza", preprocess_and_normalize),
        "3": ("Test automatici sul dataset", test_dataset),
        "4": ("Allena modello", allena_modello),
        "5": ("Valuta modello su un nuovo dataset", valuta_modello),
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
