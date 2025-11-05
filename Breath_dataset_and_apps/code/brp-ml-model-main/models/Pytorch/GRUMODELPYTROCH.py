import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class GruModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        self.gru = nn.GRU(input_size=input_shape, hidden_size=hidden_units, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_units, 32)
        self.tanh =nn.Tanh()
        self.fc2 = nn.Linear(32, output_shape)
        
        
        
        
    def forward(self, x):
        output, hidden = self.gru(x)
        x = hidden[-1]  
        
  
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        
 
        return x

    