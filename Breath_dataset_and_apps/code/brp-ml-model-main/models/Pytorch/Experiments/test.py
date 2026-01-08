import torch
import torch.nn as nn

B, T, input_dim = 32, 5, 1  # batch 32, okno 5, 1 kanał
d_model = 64

x = torch.randn(B, T, input_dim)  # przykładowy batch
embed = nn.Linear(input_dim, d_model)
x_embed = embed(x)
print(x_embed.shape)
