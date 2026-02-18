import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F


class LSTM_ATTENTION(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, dropout: float, num_layers: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_shape, out_channels=32, kernel_size=3, padding=1), # out channels is hiperparameter
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1)      
            
               
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

    def forward(self, x, return_sequence=False):
        
        x = x.permute(0,2,1) #change dimensions to fit into conv1d
        
        x = self.conv(x)
        
        x = x.permute(0,2,1)
            
        output, (h_n, c_n) = self.lstm(x)
        
        if(return_sequence):
            
            query = output
            key = output                  
            value = output   
            
            x = F.scaled_dot_product_attention(query, key, value)
            return x
        
        else:
            query = h_n[-1].unsqueeze(1)  
            key = output                  
            value = output   
                        
            x = F.scaled_dot_product_attention(query, key, value) #Attention
            
            x = x.squeeze(1)
            

            x = self.fc(x)
            return x
