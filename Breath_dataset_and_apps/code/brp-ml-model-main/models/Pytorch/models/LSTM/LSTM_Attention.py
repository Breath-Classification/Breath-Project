import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
#import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value):
        d_k = query.size(-1)
        scale = 1.0 / math.sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, value)


class LSTM_ATTENTION(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, dropout: float, num_layers: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_shape, out_channels=32, kernel_size=3, padding=1), # out channels is hiperparameter
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1)      
        )

        self.adapter = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
        )

        
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_units,
            batch_first=True,
            bidirectional =False,
            dropout =dropout,
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(hidden_units, output_shape)
        self.attention = ScaledDotProductAttention()

    def forward(self, x):
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected input with 2 or 3 dims, got shape: {x.shape}")

        
        x = x.permute(0,2,1) #change dimensions to fit into conv1d
        
        x = self.conv(x)
        
        x = x.permute(0,2,1)
            
        output, (h_n, c_n) = self.lstm(x)
        
       
        query = h_n[-1].unsqueeze(1)  
        key = output                  
        value = output   
                    
        x = self.attention(query, key, value) #Attention
        
        x = x.squeeze(1)
        x = x + self.adapter(x) # adapter specialized for a person

        x = self.fc(x)
        return x
