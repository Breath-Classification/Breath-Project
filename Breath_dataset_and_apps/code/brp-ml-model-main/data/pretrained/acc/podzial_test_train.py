# Otwieramy plik i wczytujemy wszystkie linie
with open("acc_normal_record_15-03-2024.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Obliczamy punkt podziału (np. 80%)
split_ratio = 0.8
split_index = int(len(lines) * split_ratio)

# Dzielimy dane
train_lines = lines[:split_index]
test_lines = lines[split_index:]

# Zapisujemy do dwóch plików
with open("train.txt", "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open("test.txt", "w", encoding="utf-8") as f:
    f.writelines(test_lines)

print(f"Podział zakończony! {len(train_lines)} linii treningowych, {len(test_lines)} testowych.")
