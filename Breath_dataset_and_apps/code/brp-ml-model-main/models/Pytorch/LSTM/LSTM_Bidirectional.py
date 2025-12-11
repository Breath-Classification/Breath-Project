import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F


class LSTM_BIDIRECTIONAL(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_shape,
            hidden_size=hidden_units,
            batch_first=True,
            bidirectional =True,
            dropout =0.3,
            num_layers=2
        )
        
        self.fc = nn.Linear(hidden_units*2, output_shape) #  *2 -> bidirectional forward and backward LSTM

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        
        f =h_n[-2]   #h_n has respectively bacwakrd and forward so we nedd to combine them
        x = h_n[-1]
        x = torch.cat((f,x), dim=1)
        x = self.fc(x)
        return x
