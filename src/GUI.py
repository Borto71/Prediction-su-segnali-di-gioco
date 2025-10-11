import os
import re
import subprocess
import threading
from tkinter import *
from tkinter import ttk

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





def run_extraction(selected, text_widget, progress_bar, btn_next):
    selected = listbox.get(ACTIVE)
    if not selected:
        print("Nessuna cartella selezionata!")
        return
    
    # se file pong_data_features_preprocessed_normalized exist skip extraction and preprocessing
    if os.path.exists(f"./{selected}/pong_data_features_preprocessed_normalized.csv"):
        text_widget.insert(END, f"Data for {selected} already extracted and preprocessed.\n")
        progress_bar['value'] = int(progress_bar['maximum'])
        progress_bar.update_idletasks()
        if btn_next:
            btn_next.config(state=NORMAL)
        return

    script_path = os.path.join(".", "extract_all_features.py")
    if os.path.exists(script_path):
        text_widget.insert(END, f"Estracting data from {selected}...\n")
        subprocess.run(["python", script_path])
        updatePB(progress_bar, btn_next)
        text_widget.insert(END, "Extraction completed...\n")
    else:
        print("Errore: script extract_single_game.py non trovato!")

    preprocess_script = os.path.join(".", "preprocessing.py")
    if os.path.exists(preprocess_script):
            text_widget.insert(END, "Preprocessing data...\n")
            text_widget.update()
            subprocess.run(["python", preprocess_script, selected])
            updatePB(progress_bar, btn_next)
            text_widget.insert(END, "Data preprocessed !\n")
    else:
            text_widget.insert(END, "Script normalize_data.py not found.\n")

    norm_script = os.path.join(".", "normalize_data.py")
    if os.path.exists(norm_script):
            text_widget.insert(END, "Normalizing data...\n")
            text_widget.update()
            subprocess.run(["python", norm_script, selected])
            updatePB(progress_bar, btn_next)
            text_widget.insert(END, "Data Normalized !\n")
    else:
            text_widget.insert(END, "Script normalize_data.py not found.\n")
    updatePB(progress_bar, btn_next)

def updatePB(progress_bar, btn_next=None):
    if progress_bar['value'] < progress_bar['maximum']:
        progress_bar['value'] += 10
    else:
        if btn_next:
            btn_next.config(state=NORMAL)

def updatePB(progress_bar, btn_next=None, value=None):
    print("@@@@", value, progress_bar['value'], progress_bar['maximum'])
    if value and progress_bar['value'] < progress_bar['maximum'] and value <= progress_bar['maximum']:
        progress_bar['value'] = value
        progress_bar.update_idletasks()
        if btn_next and value == progress_bar['maximum']:
            btn_next.config(state=NORMAL)


# train and test function
def run_training(epochs, batch_size, sequence_length, patience, dropout, load_model, text_output, progress_bar, btn_next):
    selected = listbox.get(ACTIVE)
    if not selected:
        print("Nessuna cartella selezionata!")
        return

    cmd = ["python", "./NN/training.py", selected, epochs, batch_size, sequence_length, patience, dropout, load_model]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True
    )

    for line in process.stdout:
        # stampa anche in console
        print(line, end="")
        
        # aggiorna la TextBox nella GUI
        text_output.insert(END, line)
        text_output.see(END)  # scroll automatico
        
        currentPatience = 0

        # aggiornamento progressivo su patience non lineare
        if line.startswith("Early stopping patience:"):
            match = re.search(r"Early stopping patience: (\d+)/(\d+)", line)
            if match:
                currentPatience = int(match.group(1)) if int(match.group(1)) > currentPatience else currentPatience
                updatePB(progress_bar, btn_next, currentPatience)

    process.wait()
    text_output.insert(END, "\n--- Training completato ---\n")
    text_output.see(END)
    





def openTrainingWindow():
    selected = listbox.get(ACTIVE)
    if not selected:
        print("Nessuna cartella selezionata!")
        return

    # nuova finestra
    new_win = Toplevel(window)
    new_win.title(f"Elaborazione - {selected}")
    new_win.geometry("500x450")
    new_win.config(bg="black")

    Label(new_win, text=f"Processing: {selected}", bg="black", fg="white", font=("Arial", 14)).pack(pady=10)

    text_output = Text(new_win, bg="white", fg="black", height=15, width=55)
    text_output.pack(padx=10, pady=10)

    ## append progress bar
    progress_bar = ttk.Progressbar(new_win, length=300, mode='determinate', maximum=30)
    progress_bar.pack(pady=10)
    progress_bar['value'] = 0  # inizializza a 0

    button_frame = Frame(new_win, bg="black")
    button_frame.pack(pady=10)

    btn_next = Button(button_frame, text="Next", command=openConfigTrainingWindow)
    btn_next.pack(side=LEFT, padx=5)
    btn_next.config(state=DISABLED)
    Button(button_frame, text="Chiudi", command=new_win.destroy).pack(side=LEFT, pady=5)


    # avvia thread separato per non bloccare la GUI
    threading.Thread(target=run_extraction, args=(selected, text_output, progress_bar, btn_next), daemon=True).start()

def openConfigTrainingWindow():
    new_win = Toplevel(window)
    new_win.geometry("500x450")
    new_win.title("Train and test the model")
    new_win.config(bg="black")

    # Frame for grid layout
    input_frame = Frame(new_win, bg="black")
    input_frame.pack(pady=30)

    # EPOCHS = 16
    # BATCH_SIZE = 16
    # SEQ_LEN = 10
    # INPUT_DIM = 12
    # NUM_CLASSES = 3
    # PATIENCE = 20
    # DROPOUT = 0.1

    sequence_length = 10
    input_dim = 9
    num_classes = 3
    patience = 20
    dropout = 0.1

    # Labels
    Label(input_frame, text="Epochs:", bg="black", fg="white", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5, sticky=E)
    Label(input_frame, text="Batch Size:", bg="black", fg="white", font=("Arial", 12)).grid(row=1, column=0, padx=5, pady=5, sticky=E)
    Label(input_frame, text="Sequence Length:", bg="black", fg="white", font=("Arial", 12)).grid(row=2, column=0, padx=5, pady=5, sticky=E)

    Label(input_frame, text="Patience:", bg="black", fg="white", font=("Arial", 12)).grid(row=5, column=0, padx=5, pady=5, sticky=E)
    Label(input_frame, text="Dropout:", bg="black", fg="white", font=("Arial", 12)).grid(row=6, column=0, padx=5, pady=5, sticky=E)

    # Default values
    entry_epochs = Entry(input_frame, width=10)
    entry_epochs.insert(0, "50")
    entry_epochs.grid(row=0, column=1, padx=5, pady=5)
    entry_epochs.config(state=NORMAL)  # Ensure entry is enabled
    entry_epochs.focus_set()           # Set focus for immediate input

    entry_batch = Entry(input_frame, width=10)
    entry_batch.insert(0, "32")
    entry_batch.grid(row=1, column=1, padx=5, pady=5)

    entry_seq_len = Entry(input_frame, width=10)
    entry_seq_len.insert(0, str(sequence_length))
    entry_seq_len.grid(row=2, column=1, padx=5, pady=5)

    entry_patience = Entry(input_frame, width=10)
    entry_patience.insert(0, str(patience))
    entry_patience.grid(row=5, column=1, padx=5, pady=5)

    entry_dropout = Entry(input_frame, width=10)
    entry_dropout.insert(0, str(dropout))
    entry_dropout.grid(row=6, column=1, padx=5, pady=5)


    button_frame2 = Frame(new_win, bg="black")
    button_frame2.pack(pady=10)

    newExtractDataButton = Button(button_frame2, text="Avvia il training", command=lambda: OutputTrainingWindow(
            entry_epochs.get(),
            entry_batch.get(),
            entry_seq_len.get(),
            entry_patience.get(),
            entry_dropout.get(),
            "False"
        ))
    newExtractDataButton.pack(side=LEFT, padx=5)

    resumeExtractDataButton = Button(button_frame2, text="Riprendi il training", command=lambda: OutputTrainingWindow(
            entry_epochs.get(),
            entry_batch.get(),
            entry_seq_len.get(),
            entry_patience.get(),
            entry_dropout.get(),
            "True"
    ))

    resumeExtractDataButton.pack(side=LEFT, padx=5)

    exitButton = Button(button_frame2, text="Esci", command=new_win.quit)
    exitButton.pack(side=LEFT, padx=5)




def OutputTrainingWindow(epochs, batch_size, sequence_length, patience, dropout, load_model):
    selected = listbox.get(ACTIVE)
    if not selected:
        print("Nessuna cartella selezionata!")
        return
    # nuova finestra
    new_win = Toplevel(window)
    new_win.title(f"Training Output")
    new_win.geometry("1500x1000")
    new_win.config(bg="black")

    Label(new_win, text=f"Processing: {selected}", bg="black", fg="white", font=("Arial", 14)).pack(pady=10)

    text_output = Text(new_win, bg="white", fg="black", height=50, width=200)
    text_output.pack(padx=10, pady=10)

    ## append progress bar
    progress_bar = ttk.Progressbar(new_win, length=300, mode='determinate', maximum=patience)
    progress_bar.pack(pady=10)
    progress_bar['value'] = 0

    button_frame = Frame(new_win, bg="black")
    button_frame.pack(pady=10)

    btn_next = Button(button_frame, text="Close", command=new_win.destroy)
    btn_next.pack(side=LEFT, padx=5)
    btn_next.config(state=DISABLED)


    # avvia thread separato per non bloccare la GUI
    threading.Thread(target=run_training, args=(epochs, batch_size, sequence_length, patience, dropout, load_model, text_output, progress_bar, btn_next), daemon=True).start()



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
