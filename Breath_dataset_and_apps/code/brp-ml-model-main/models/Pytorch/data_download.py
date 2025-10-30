import torch
from torch.utils.data import Dataset
import numpy as np
from keras.src.utils import to_categorical

class SequenceDataset(Dataset):
    def __init__(self, filename, expand_dims=True, convert_to_categorical=False):
        """
        Wczytuje dane z pliku txt i przygotowuje tensory.

        filename: ścieżka do pliku z danymi treningowymi
        sensor_type: typ sensora, używany do znalezienia pliku testowego
        expand_dims: czy dodać dodatkowy wymiar (np. dla GRU)
        convert_to_categorical: czy zamienić etykiety na one-hot (dla Keras-style modeli)
        """

        # 1️⃣ Wczytaj dane treningowe
        train_sequences = []
        with open(filename, "r") as f:
            for line in f.readlines():
                train_sequences.append([float(value) for value in line.strip().split(",")])
        data = np.array(train_sequences)

        

        # 3️⃣ Rozdziel X i y
        self.X = data[:, :-1]
        self.y = data[:, -1]
      
        # 4️⃣ Rozszerz wymiar (jeśli np. model oczekuje [batch, channels, seq_len])
        if expand_dims:
            self.X = np.expand_dims(self.X, axis=1)
          
        # 5️⃣ Zamiana na one-hot (opcjonalnie)
        if convert_to_categorical:
            num_classes = len(np.unique(self.y))
            self.y = to_categorical(self.y, num_classes=num_classes)
           

        # 6️⃣ Zamiana na tensory PyTorch
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long if not convert_to_categorical else torch.float32)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
