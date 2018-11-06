"""A minimal sample script for illustration of basic usage of NCE module"""

import torch
from nce import IndexLinear

class_freq = [1, 2, 3, 4, 5, 6, 7]  # an unigram class probability
freq_count = torch.FloatTensor(class_freq)
noise = freq_count / freq_count.sum()

nce_linear = IndexLinear(
    embedding_dim=100,  # input dim
    num_classes=300000,  # output dim
    noise=noise,
)

input = torch.Tensor(200, 100)
target = torch.ones(200, 1).long()
# training mode
loss = nce_linear(target, input).mean()
print(loss.item())

# evaluation mode for fast probability computation
nce_linear.eval()
prob = nce_linear(target, input).mean()
print(prob.item())
