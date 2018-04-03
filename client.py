import torch
from torch import nn
from torch import distributed

class Net(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(5,5)
        self.emb = nn.Embedding(5,5)

distributed.init_process_group(backend='mpi', world_size=0, rank=0)
print('initialized, client')
a = torch.zeros(5, 5)
distributed.send(a, 0)
