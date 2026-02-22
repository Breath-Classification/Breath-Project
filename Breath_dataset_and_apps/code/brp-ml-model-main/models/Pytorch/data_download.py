import torch
from torch.utils.data import Dataset
import numpy as np
from keras.src.utils import to_categorical

'''
 The purpose of this code is to read data from a file containing accelerometer/tensometr measurements
 and convert them into tensors.
 SequenceDataset loads the data sequentially — each reading X is assigned a specific class y.
 BlockDataset loads the data in blocks — readings are grouped into blocks using a sliding window approach,
 and each block is assigned the class of its last element.
 Information from the first and second derivatives of the signal is added to the data.
 There is also an option to add Gaussian noise.
'''

class SequenceDataset(Dataset): #Data loaded sequentially
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
           
        #Convert data to tensor
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long if not convert_to_categorical else torch.float32)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
class BlockDataset(Dataset): #Sliding window
    def __init__(self, filename, block_size, convert_to_categorical=False, sigma=0.01, augment=True):
        
        self.sigma = sigma
        self.augment = augment

        data = []
        with open(filename, "r") as f:
            for line in f.readlines():
                data.append([float(v) for v in line.strip().split(",")])
        data = np.array(data)

        X = data[:, :-1]
        y = data[:, -1]

        
        blocks_X, blocks_y = [], []  #Create blocks
        for i in range(len(X) - block_size + 1):
            blocks_X.append(X[i:i + block_size])
            blocks_y.append(y[i + block_size - 1]) 

        self.X = np.array(blocks_X)  
        self.y = np.array(blocks_y)

        if convert_to_categorical:
            num_classes = len(np.unique(self.y))
            self.y = to_categorical(self.y, num_classes=num_classes)

        #Convert data to tensor
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long if not convert_to_categorical else torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        '''
        #first derivative dx[t] =x[t]-x[t-1]
        dx = torch.zeros_like(x)
        dx[1:]=x[1:]-x[:-1]
        x=torch.cat([x,dx],dim=1) #[30,6] -> [30,12]

        #second derivative ddx[t] = x[t+1] - 2x[t] +x[t-1]
        ddx = torch.zeros_like(x)
        ddx[1:-1] = x[2:] - 2 * x[1:-1] + x[:-2]
        x = torch.cat([x,ddx], dim=1)  #[30,12] -> [30,18]
        '''
        
        #add Gausian noise
        if self.augment:
            noise = torch.randn_like(x) * self.sigma
            x = x + noise

        return x, y
    
