import torch
from torch.nn import Parameter, CrossEntropyLoss
import torch.nn.functional as F
from torch.autograd import Variable

from nce import NCELoss
from basis_module import BasisModule

class IndexBasisLinear(NCELoss):
    """A module to partially decode every basis in embedding chunks

    Args:
        - in_features: size of each input sample
        - out_features: size of vocabulary
        - num_clusters: number of clusters per basis
        - bias: same as nn.Linear, whether to use bias
        - num_basis: number of chunks that should be normalized at parallel

    Shape:
        - Input: (N, in\_features)
        - intermediate Output: (Nb, N, num_clusters)
        - Output: (N, V)

    Attributes:
        - bias: trainable bias in shape (num_clusters)
    """

    def __init__(self, in_features, out_features, num_basis, num_clusters, *args, **kwargs):
        super(IndexBasisLinear, self).__init__(*args, **kwargs)
        self.bm = BasisModule(out_features, in_features, num_basis, num_clusters)
        self.ce = CrossEntropyLoss(reduce=False)

        # get integer in python3
        self.features_per_basis = in_features // num_basis
        self.bias = Parameter(0.01 * torch.randn(out_features))

    def enable_basis(self):
        self.bm.enable_basis()

    def disable_basis(self):
        self.bm.disable_basis()

    def basis_mode(self, basis):
        self.bm.basis_mode(basis)

    def get_score(self, target_idx, noise_idx, input):
        """
        Shape:
            - target_batch :math:`(N, E, 1+N_r)`where `N = length, E = embedding size, N_r = noise ratio`
        """

        # flatten the following matrix
        input = input.contiguous().view(-1, input.size(-1))
        original_size = target_idx.size() # the size will be used to pack the output of indexlinear
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx.view(-1, noise_idx.size(-1))

        indices = torch.cat([target_idx.unsqueeze(-1), noise_idx], dim=-1)

        # the pytorch's [] operator can't BP correctly with redundant indices
        # before version 0.2.0
        input = input.unsqueeze(1)
        if self.bm.basis:
            target_batch = self.bm.pq.get_centroid(indices.view(-1).data)
        else:
            target_batch = self.bm.original_matrix[indices.view(-1)]
        target_batch = target_batch.view(*indices.size(), -1).transpose(1,2)
        bias = self.bias.index_select(0, indices.view(-1)).view_as(indices).unsqueeze(1)
        out = torch.baddbmm(1, bias, 1, input, target_batch).view(*original_size, -1)
        target_score, noise_score = out[:, :, 0], out[:, :, 1:]
        return target_score, noise_score

    def ce_loss(self, target_idx, input):
        if self.bm.basis:
            weight = self.bm.pq.get_centroid()
        else:
            weight = self.bm.original_matrix
        score = F.linear(input, weight, self.bias) # (N, V)
        loss = self.ce(score.view(-1, score.size(-1)), target_idx.view(-1)).view_as(target_idx)
        return loss
