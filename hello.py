import torch.nn
import torch
x = torch.rand(32,96,7)
print(x)
embed = torch.nn.Embedding(100, 64)