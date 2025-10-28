import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, filename, expand_dims=True):
        """
        Wczytuje dane z pliku txt i przygotowuje tensory.

        filename: ścieżka do pliku z danymi
        expand_dims: czy dodać dodatkowy wymiar dla GRU
        """
        # 1️⃣ Wczytaj dane z pliku
        sequences = []
        with open(filename, "r") as f:
            for line in f.readlines():
                sequences.append([float(value) for value in line.strip().split(",")])
        data = np.array(sequences)

        # 2️⃣ Rozdziel X i y
        self.X = data[:, :-1]
        self.y = data[:, -1].astype(int)  # upewnij się, że y jest int dla CrossEntropyLoss

        # 3️⃣ Jeśli expand_dims=True, dodaj wymiar
        if expand_dims:
            self.X = np.expand_dims(self.X, axis=1)

        # 4️⃣ Zamień na tensory PyTorch
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
        return self.X, self.y

    def __len__(self):
        # 5️⃣ Ile jest próbek?
        return len(self.X)

    def __getitem__(self, idx):
        # 6️⃣ Co zwraca pojedynczy przykład?
        return self.X[idx], self.y[idx]