import torch
from torch import nn
import torch.nn.functional as F

class Seq2SeqGRU(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.gru = nn.GRU(input_size=input_shape, hidden_size=hidden_units, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_units, output_shape)
    def forward(self, x):
        output, hidden = self.gru(x)
        x = self.relu(output)
        x = self.dropout1(x)
        x = self.fc(x)
        return x