import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

class LSTM_BASE(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_shape,
            hidden_size=hidden_units,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_units, output_shape)

    def forward(self, x, return_sequence=False):
        output, (h_n, c_n) = self.lstm(x)
        
        if(return_sequence):
            return output
        x= h_n[-1]

        x = self.fc(x)
        return x
