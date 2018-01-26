import torch
import torch.nn as nn
from torch.nn import Parameter

from product_quantizer import ProductQuantizer as PQ

class BasisModule(nn.Module):
    """A wrapper class using PQ to reduce parameters

    The basis module utilize product quantization as an efficient quantizer
    for parameter reduction.

    Arguments:
        - num_samples: number of samples (feature vectors)
        - dimension: feature dimension size
        - num_sub: number of sub-quantizer
        - num_clusters: the number of clusters in each sub-quantizer
        - basis: basis mode switch

    Shape:
        - Input: (B, N) indices of words
        - Output: (B, N, embedding_dim)
    """

    def __init__(self, num_samples, dimension, num_sub=2, num_clusters=400):
        super(BasisModule, self).__init__()
        self.num_samples = num_samples
        self.dimension = dimension
        self.num_sub = num_sub
        self.num_clusters = num_clusters

        self.pq = PQ(dimension, num_sub, num_clusters)
        init_weight = torch.zeros(num_samples, dimension).uniform_(-0.1, 0.1)
        self.original_matrix = Parameter(init_weight)

        self.basis = False


    def forward(self, input):
        raise NotImplementedError('The basis module is not forwardable')

    def enable_basis(self):
        """Enable the basis mode as approximation"""
        if not self.basis:
            self.basis = True
            self.pq.train_code(self.original_matrix.data)

    def disable_basis(self):
        """Disable the basis mode"""
        if self.basis:
            self.basis = False
            self.original_matrix = Parameter(self.pq.get_centroid().data) # all centroids

    def basis_mode(self, basis):
        if basis:
            self.enable_basis()
        else:
            self.disable_basis()
