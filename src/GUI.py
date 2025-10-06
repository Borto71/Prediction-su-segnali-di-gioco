import os
import subprocess
from tkinter import *

# main window
window = Tk()
window.geometry("420x420")
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
    
    script_path = os.path.join(".", "extract_single_game.py")
    if os.path.exists(script_path):
        print(f"Estrazione dati da {selected}...")
        subprocess.run(["python", script_path, selected])
        print("Estrazione completata!")
    else:
        print("Errore: script extract_single_game.py non trovato!")

# buttons
newGameButton = Button(window, text="Play a new game", command=play_new_game)
newGameButton.pack(pady=10)

extractDataButton = Button(window, text = "Process the selected game_data", command = extract_selected_game_data)
extractDataButton.pack(pady=10)

# listbox
listbox = Listbox(window, bg="azure", fg="black", font=("Arial", 12))
listbox.pack(padx=20, pady=20, fill=BOTH, expand=True)

# layout


# initial render
update_listbox()

window.mainloop()
