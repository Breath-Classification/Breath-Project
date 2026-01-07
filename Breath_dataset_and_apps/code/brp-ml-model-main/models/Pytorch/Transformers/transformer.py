import torch
import torch.nn as nn


class SinPositionalEncoding(nn.Module):
    def __init__(self):
        super.__init__()
        
    def forward(x):
        return x

class Transformer(nn.Module):
    def __init__(self):
        super.__init__()
        
        # 1. self.embed ->embeding
        # 2. positional -> okresl pozycje
        # 3. główny transformer Att is all you need 
        self.encoder = nn.TransformerEncoder( ) # parametry layers = TransformerEncoderLayers i inne
        # 4. wyjscie transformera to (B,T,32)
        # 5. z wyjascia zrob polling  a) CLS token b) mean pooling
        # 6. klasyfikator self.linear  
    def forward(x):
        return x