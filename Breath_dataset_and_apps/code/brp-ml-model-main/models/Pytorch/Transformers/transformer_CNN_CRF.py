import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchcrf import CRF
import math


"https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308"
class SinPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer_CNN_CRF(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, d_model: int, dropout :float =0.1, num_layrer:int =2, dim_feedforward:int =64, nhead : int =2):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_shape, out_channels=32, kernel_size=3, padding=1), # out channels is hiperparameter
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # out channels is hiperparameter
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1)         
        )
    
        
        self.embed = nn.Linear(64,d_model)
        # 1. self.embed ->embeding
        
        self.position = SinPositionalEncoding(d_model)
        # 2. positional -> okresl pozycje
        
        
        # 3. główny transformer Att is all you need 
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layrer) # parametry layers = TransformerEncoderLayers i inne
        
        
        # 4. wyjscie transformera to (B,T,32)
        # 5. z wyjascia zrob polling  a) CLS token b) mean pooling
        # 6. klasyfikator self.linear  
        
        self.crf = CRF(output_shape)
        
        self.fc = nn.Linear(d_model, output_shape)

    def forward(self, x):
        
        x = x.permute(0,2,1) #change dimensions to fit into conv1d
        
        x = self.conv(x)
        
        x = x.permute(0,2,1)
            
        x=self.embed(x)
        x=self.position(x)
        x=self.encoder(x)
        

        x = self.fc(x)
        return x