import torch
import torch.nn as nn
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

class Transformer(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, d_model: int):
        super().__init__()
        
        self.embed = nn.Linear(input_shape,d_model)
        # 1. self.embed ->embeding
        
        self.position = SinPositionalEncoding(d_model)
        # 2. positional -> okresl pozycje
        
        
        # 3. główny transformer Att is all you need 
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2,dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2) # parametry layers = TransformerEncoderLayers i inne
        
        
        # 4. wyjscie transformera to (B,T,32)
        # 5. z wyjascia zrob polling  a) CLS token b) mean pooling
        # 6. klasyfikator self.linear  
        self.fc = nn.Linear(d_model, output_shape)
    def forward(self,x):
        x=self.embed(x)
        x=self.position(x)
        x=self.encoder(x)
        
        x=x.mean(dim=1) #pooling
        x=self.fc(x)
        return x