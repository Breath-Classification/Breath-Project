import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F


class LSTM_STACKED(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_shape,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=3, # number of layers
            dropout=0.3
        )
        
        self.fc = nn.Linear(hidden_units, output_shape)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        
        x = h_n[-1]

        x = self.fc(x)
        return x
