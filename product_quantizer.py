import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter


def basis_cluster(weight, num_basis, num_clusters, cuda=False):
    """Divide the weight into `num_basis` basis and clustering

    Params:
        - weight: weight matrix to do basis clustering
        - num_basis: number of basis, also the dimension of coordinates
        - num_cluster: number of clusters per basis

    Return:
        - basis: (Nb, Nc, E/Nb)the cluster centers for each basis.
        - coordinates: (V, Nb) the belongings for basis of each token.
    """
    partial_embeddings = weight.chunk(num_basis, dim=1)

    coordinates = []
    basis = []
    if not cuda:
        from sklearn.cluster import KMeans
        clustor = KMeans(init='k-means++', n_clusters=num_clusters, n_init=1)
    for partial_embedding in partial_embeddings:
        if cuda:
            from libKMCUDA import kmeans_cuda
            centroid, coordinate = kmeans_cuda(partial_embedding.numpy(), num_clusters, seed=7)
            # some clusters may have zero elements, thus the centroids becomes [nan] in libKMCUDA
            centroid = np.nan_to_num(centroid)
        else:
            clustor.fit(partial_embedding.numpy())
            centroid, coordinate = clustor.cluster_centers_, clustor.labels_
        basis.append(torch.from_numpy(centroid.astype('float')))
        coordinates.append(torch.from_numpy(coordinate.astype('int32')))

    basis = torch.stack(basis).float() # Nb X Nc(clusters) X E/Nb
    coordinates = torch.stack(coordinates).t().long() # V X Nb(number of basis)
    return basis, coordinates


class ProductQuantizer(nn.Module):
    """Product Quantizer for pytorch

    The API simply follows the ProductQuantizer in `faiss` project

    Parameters:
        - dimension: dimensionality of the input vectors
        - num_sub: L number of sub-quantizers (M)
        - k: number of clusters per sub-vector index (2^^nbits)

    Attributes:
        - codebook: (V, Nb) the coordinates of words under specific basis
        - centroid: (Nb, Nc, E/Nb)the cluster centroids of original embedding matrix


    """

    def __init__(self, dimension, num_sub, k):
        super(ProductQuantizer, self).__init__()
        self.dimension = dimension
        self.num_sub = num_sub
        self.k = k
        if not dimension % num_sub == 0:
            raise ValueError('Embedding size({}) should be '
                             'divisible by basis number({})'.format(dimension, num_sub))

    def train_code(self, data_matrix):
        """Get the codebook and centroids from data

        Args:
            - data_matrix: (N, D) where N is number of vectors
            D is dimension

        Returns:
            None
        """

        centroid, codebook = basis_cluster(data_matrix.cpu(), self.num_sub, self.k)
        self.centroid = Parameter(centroid)
        self.register_buffer('codebook', Variable(codebook))
        if data_matrix.is_cuda:
            self.cuda()

    def get_centroid(self, index=None):
        """Get the reproduction value for training data

        Args:
            index: (C) the index(es) to look-up
        """
        if index is None:
            code = self.codebook
        else:
            code = self.codebook[index]
        return self.decode(code)

    def decode(self, code):
        """Decode the code into reproduction value

        The reproduction value is a concatenation of centroids in
        different sub-quantizer.

        Args:
            - code: (C, N_s) where C is arbitrary number of codes
            N_s is number of sub-quantizers

        Return:
            - centroid: (C, D) the reproduction values of input codes
        """
        sub_centroids = []
        for cur_sub in range(self.num_sub):
            cur_index = code[:, cur_sub]
            sub_centroid = self.centroid[cur_sub][cur_index] # N X E/Nb
            sub_centroids.append(sub_centroid)

        centroid = torch.cat(sub_centroids, dim=1)
        return centroid


    def compute_code(self, point):
        pass
