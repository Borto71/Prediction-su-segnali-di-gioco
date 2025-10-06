import os
import subprocess
from tkinter import *

cartella_scelta = ""

# main window
window = Tk()
window.geometry("450x450")
window.title("Prediction game signals")
window.config(background="black")

# fetch folders
def get_game_data_folders():
    src_path = "./"
    folders = []
    if os.path.exists(src_path):
        for name in os.listdir(src_path):
            path = os.path.join(src_path, name)
            if os.path.isdir(path) and name.startswith("game_data_"):
                folders.append(name)
    return sorted(folders)

# update listbox
def update_listbox():
    listbox.delete(0, END)
    for folder in get_game_data_folders():
        listbox.insert(END, folder)

# play new game func
def play_new_game():
    script_path = os.path.join(".", "play_pong.py")
    if os.path.exists(script_path):
        #play and update list
        subprocess.run(["python", script_path])
        update_listbox()
    else:
        print("Errore: file play_pong.py non trovato!")

def extract_selected_game_data():
    selected = listbox.get(ACTIVE)
    if not selected:
        print("Nessuna cartella selezionata!")
        return
    
    script_path = os.path.join(".", "extract_all_features.py")
    if os.path.exists(script_path):
        print(f"Estrazione dati da {selected}...")
        subprocess.run(["python", script_path])
        print("Estrazione completata!")
    else:
        print("Errore: script extract_single_game.py non trovato!")

    script_path = os.path.join(".", "normalize_data.py")

def on_select(event):
    global cartella_scelta
    selected = listbox.get(ACTIVE)
    cartella_scelta = selected
    print(f"Cartella selezionata: {cartella_scelta}")

# label
label = Label(
    window,
    text="Seleziona una cartella di dati di gioco o avvia una nuova partita:",
    bg="black",
    fg="white",
    font=("Arial", 14),
    wraplength=400,
    justify="center"
)
label.pack(pady=10)

# play new game Button
newGameButton = Button(window, text="Play a new game", command=play_new_game)
newGameButton.pack(pady=10)

# listbox
listbox = Listbox(window, bg="azure", fg="black", font=("Arial", 12))
listbox.pack(padx=20, pady=20, fill=BOTH, expand=True)
listbox.bind("<<ListboxSelect>>", on_select)

# extract selected game_data Button
button_frame = Frame(window, bg="black")
button_frame.pack(pady=10)

extractDataButton = Button(button_frame, text="Processa i dati di gioco selezionati", command=extract_selected_game_data)
extractDataButton.pack(side=LEFT, padx=5)

exitButton = Button(button_frame, text="Esci", command=window.quit)
exitButton.pack(side=LEFT, padx=5)

# initial render
update_listbox()


window.mainloop()
