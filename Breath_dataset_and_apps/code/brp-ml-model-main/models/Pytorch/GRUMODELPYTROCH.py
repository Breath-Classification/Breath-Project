import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

class GruModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        self.gru = nn.GRU(input_size=input_shape, hidden_size=hidden_units, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_units, 32)
        self.tanh =nn.Tanh()
        self.fc2 = nn.Linear(32, output_shape)
        
        
    def forward(self, x):
        output, hidden = self.gru(x)
        x = hidden[-1]  
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

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


#https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class GRUAttentionModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.dropout = nn.Dropout(0.03)
        self.gru = nn.GRU(input_size=input_shape, hidden_size=hidden_units, batch_first=True)
        self.dropout = nn.Dropout(0.07)
        self.attn = BahdanauAttention(hidden_units)
        self.fc = nn.Linear(hidden_units, output_shape)
    def forward(self, x):
        x=self.dropout(x)
        outputs, hidden = self.gru(x)
        query = hidden.permute(1, 0, 2)
        context, weights = self.attn(query, outputs)
        y_hat = self.fc(context.squeeze(1))
        return y_hat