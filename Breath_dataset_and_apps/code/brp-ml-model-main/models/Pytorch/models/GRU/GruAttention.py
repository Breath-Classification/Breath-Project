import torch
from torch import nn
import torch.nn.functional as F

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