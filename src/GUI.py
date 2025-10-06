import os
import subprocess
import threading
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



#
def run_extraction(selected, text_widget):
    selected = listbox.get(ACTIVE)
    if not selected:
        print("Nessuna cartella selezionata!")
        return
    
    script_path = os.path.join(".", "extract_all_features.py")
    if os.path.exists(script_path):
        text_widget.insert(END, f"Estracting data from {selected}...\n")
        subprocess.run(["python", script_path])
        text_widget.insert(END, "Extraction completed...\n")
    else:
        print("Errore: script extract_single_game.py non trovato!")

    preprocess_script = os.path.join(".", "preprocessing.py")
    if os.path.exists(preprocess_script):
            text_widget.insert(END, "Preprocessing data...\n")
            text_widget.update()
            subprocess.run(["python", preprocess_script, selected])
            text_widget.insert(END, "Data preprocessed !\n")
    else:
            text_widget.insert(END, "Script normalize_data.py not found.\n")

    norm_script = os.path.join(".", "normalize_data.py")
    if os.path.exists(norm_script):
            text_widget.insert(END, "Normalizing data...\n")
            text_widget.update()
            subprocess.run(["python", norm_script, selected])
            text_widget.insert(END, "Data Normalized !\n")
    else:
            text_widget.insert(END, "Script normalize_data.py not found.\n")

def openTrainingWindow():
    selected = listbox.get(ACTIVE)
    if not selected:
        print("Nessuna cartella selezionata!")
        return

    # nuova finestra
    new_win = Toplevel(window)
    new_win.title(f"Elaborazione - {selected}")
    new_win.geometry("500x400")
    new_win.config(bg="black")

    Label(new_win, text=f"Processing: {selected}", bg="black", fg="white", font=("Arial", 14)).pack(pady=10)

    text_output = Text(new_win, bg="white", fg="black", height=15, width=55)
    text_output.pack(padx=10, pady=10)

    Button(new_win, text="Chiudi", command=new_win.destroy).pack(pady=5)

    # avvia thread separato per non bloccare la GUI
    threading.Thread(target=run_extraction, args=(selected, text_output), daemon=True).start()

    

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

# buttons + Frame
button_frame = Frame(window, bg="black")
button_frame.pack(pady=10)

extractDataButton = Button(button_frame, text="Processa i dati di gioco selezionati", command=openTrainingWindow)
extractDataButton.pack(side=LEFT, padx=5)

exitButton = Button(button_frame, text="Esci", command=window.quit)
exitButton.pack(side=LEFT, padx=5)

# initial update
update_listbox()


window.mainloop()
