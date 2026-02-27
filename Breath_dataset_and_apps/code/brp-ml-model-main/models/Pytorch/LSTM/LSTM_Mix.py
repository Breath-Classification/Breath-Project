import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchcrf

class LSTM_Mix(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, dropout : float =0.0, num_layers : int =1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_shape, out_channels=32, kernel_size=3, padding=1), # out channels is hiperparameter
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # out channels is hiperparameter
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1)         
        )
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_units,
            batch_first=True,
            bidirectional =True,
            dropout =dropout,
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(hidden_units*2, output_shape)

    def forward(self, x):
        
        x = x.permute(0,2,1) #change dimensions to fit into conv1d
        
        x = self.conv(x)
        
        x = x.permute(0,2,1)
            
        output, (h_n, c_n) = self.lstm(x)
    
        x = output

        x = self.fc(x)
        return x