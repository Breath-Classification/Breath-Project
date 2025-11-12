import torch
from torch.utils.data import Dataset
import numpy as np
from keras.src.utils import to_categorical

class SequenceDataset(Dataset): #dane ladowane sekwencyjnie
    def __init__(self, filename, expand_dims=True, convert_to_categorical=False):
       
        train_sequences = []
        with open(filename, "r") as f:
            for line in f.readlines():
                train_sequences.append([float(value) for value in line.strip().split(",")])
        data = np.array(train_sequences)

        
        self.X = data[:, :-1]
        self.y = data[:, -1]
      
        if expand_dims:
            self.X = np.expand_dims(self.X, axis=1)
          
        if convert_to_categorical:
            num_classes = len(np.unique(self.y))
            self.y = to_categorical(self.y, num_classes=num_classes)
           

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long if not convert_to_categorical else torch.float32)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
class BlockDataset(Dataset): #dane ladowane blokowo
    def __init__(self, filename, block_size, convert_to_categorical=False):
        
        data = []
        with open(filename, "r") as f:
            for line in f.readlines():
                data.append([float(v) for v in line.strip().split(",")])
        data = np.array(data)

        X = data[:, :-1]
        y = data[:, -1]

        
        blocks_X, blocks_y = [], []  #tworzenie blokow
        for i in range(len(X) - block_size + 1):
            blocks_X.append(X[i:i + block_size])
            blocks_y.append(y[i + block_size - 1]) #etykieta bloku taka jak ostatni element 

        self.X = np.array(blocks_X)  
        self.y = np.array(blocks_y)

        if convert_to_categorical:
            num_classes = len(np.unique(self.y))
            self.y = to_categorical(self.y, num_classes=num_classes)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long if not convert_to_categorical else torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class BlockDatasetHalf(Dataset): #dane ladowane blokowo
    def __init__(self, filename, block_size, convert_to_categorical=False):
        
        data = []
        with open(filename, "r") as f:
            for line in f.readlines():
                data.append([float(v) for v in line.strip().split(",")])
        data = np.array(data)

        X = data[:, :-1]
        y = data[:, -1]

        
        blocks_X, blocks_y = [], []  #tworzenie blokow
        for i in range(len(X) - block_size + 1):
            blocks_X.append(X[i:i + block_size])
            blocks_y.append(y[i + (block_size - 1)//2]) #etykieta bloku taka jak srodkowy element 

        self.X = np.array(blocks_X)  
        self.y = np.array(blocks_y)

        if convert_to_categorical:
            num_classes = len(np.unique(self.y))
            self.y = to_categorical(self.y, num_classes=num_classes)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long if not convert_to_categorical else torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BlockTargetDataset(Dataset): #dane ladowane blokowo i z wykorzystaniem wielu y
    def __init__(self, filename, block_size, target_size, convert_to_categorical=False):
        
        data = []
        with open(filename, "r") as f:
            for line in f.readlines():
                data.append([float(v) for v in line.strip().split(",")])
        data = np.array(data)

        X = data[:, :-1]
        y = data[:, -1]

        
        blocks_X, blocks_y = [], []  #tworzenie blokow
        for i in range(len(X) - block_size -target_size+ 1):
            blocks_X.append(X[i:i + block_size])
            blocks_y.append(y[i + block_size - 1 : i+ block_size+target_size]) #tyle etykiet ile wynosi target

        self.X = np.array(blocks_X)  
        self.y = np.array(blocks_y)

        if convert_to_categorical:
            num_classes = len(np.unique(self.y))
            self.y = to_categorical(self.y, num_classes=num_classes)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long if not convert_to_categorical else torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]