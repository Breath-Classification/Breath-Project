import os
from pathlib import Path
from scripts.load_data import load_tagged_data, save_sequences_to_concatenated

# ==============================
# KONFIGURACJA
# ==============================
PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FOLDER = PROJECT_DIR / "data" / "NewData"        # folder z plikami txt
SEQUENCE_FOLDER = INPUT_FOLDER / "sequence"  # folder do sekwencji
WINDOW_SIZE = 5                       # tensometr = 5, akcelerometr = 11

# Utwórz folder sequence jeśli nie istnieje
os.makedirs(SEQUENCE_FOLDER, exist_ok=True)

# ==============================
# Tworzenie sekwencji dla każdego pliku
# ==============================
txt_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]

def save_sequences(
    file_to_retrieve_sequences: str, file_to_save: str, size: int, certain_tags: list[int] | None = None,
) -> None:
    with open(file_to_retrieve_sequences):
        data, tags = load_tagged_data(file_to_retrieve_sequences)
    sequences = []
    for i in range(size, len(data)):
        position = i - size + size//2
        sequence = data[i - size: i]
        sequence.append(abs(max(sequence) - min(sequence)))
        if certain_tags is not None:
            if tags[position] in certain_tags:
                sequences.append(sequence + [tags[position] + 1])
        else:
            if tags[position] != 999.0:
                sequences.append(sequence + [tags[position] + 1])

    with open(file_to_save, "w") as file:
        for seq in sequences:
            file.write(",".join(map(str, seq)) + "\n")



for txt_file in txt_files:
    input_file = INPUT_FOLDER / txt_file
    output_file = SEQUENCE_FOLDER / txt_file
    
    # Wczytaj i utwórz sekwencje
    save_sequences(
        file_to_retrieve_sequences=input_file,
        file_to_save=output_file,
        size=WINDOW_SIZE,
        certain_tags=None  # zachowaj wszystkie etykiety
    )
    print(f"Stworzono sekwencje dla: {txt_file}")

# ==============================
# Tworzenie jednego pliku concatenated
# ==============================
concatenated_file = SEQUENCE_FOLDER / "concatenated.txt"

# Czyść plik concatenated
with open(concatenated_file, "w"):
    pass

for seq_file in os.listdir(SEQUENCE_FOLDER):
    if seq_file.endswith(".txt") and seq_file != "concatenated.txt":
        save_sequences_to_concatenated(
            file_to_retrieve_sequences=SEQUENCE_FOLDER / seq_file,
            file_to_save=concatenated_file
        )

print(f"Plik concatenated utworzony: {concatenated_file}")
