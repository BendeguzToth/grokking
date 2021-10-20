"""
2-layer transformer model for experiments described in (Power et al., 2021).

https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf
"""

import torch
from torch import Tensor, tensor
from torch.nn import Embedding, Module, TransformerEncoderLayer, TransformerEncoder
import numpy as np


def make_triu(m: int, device: torch.device):
    a = torch.full(size=(m, m), fill_value=False, device=device)
    a[np.triu_indices(m, 1)] = True
    return a


class Grokformer(Module):
    """
    Small transformer for grokking. It takes four tokens: <x> <o> <y> <=> and
    produces a result. Only returns the last prediction for each sequence in the batch.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device):
        super(Grokformer, self).__init__()
        self.embedding = Embedding(num_embeddings=num_embeddings + 4, embedding_dim=embedding_dim, device=device)
        # [0, num_embeddings] is the symbol embeddings, [num_embeddings + 1, num_embeddings + 4] is positional encodings.
        self.n_embeddings = num_embeddings + 4
        layer = TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, dropout=0., batch_first=True)
        self.network = TransformerEncoder(encoder_layer=layer, num_layers=2)
        self.network.to(device)
        # Used to index into the last five elements of self.embedding for the positional encodings.
        self.pos_idx = tensor([num_embeddings + k for k in range(4)], device=device)
        self.device = device
        self.mask = make_triu(4, self.device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the network.
        :param x: An int tensor of shape (n, 4).
        :return: A tensor of shape (n, num_embeddings).
        """
        n, _ = x.shape
        pos = self.embedding(self.pos_idx)  # pos: (4, embedding_dim)
        pos = pos.view((1, -1))             # pos: (1, 4*embedding_dim)
        pos = pos.repeat((n, 1))            # pos: (n, 4*embedding_dim)

        emb = self.embedding(x)             # emb: (n, 4, embedding_dim)
        emb = emb.view((n, -1))             # emb: (n, 4*embedding_dim)

        # 'memory' is the sequence from the encoder. Since this is a decoder-only
        # architecture, the encoder output will be replaced by the input sequence
        # <x> <o> <y> <=> <xoy>
        src = emb + pos                  # src: (n, 4*embedding_dim)
        src = src.view((n, 4, -1))       # src: (n, 4, embedding_dim)

        # Feeding it through the transformer.
        res = self.network.forward(src=src, mask=self.mask)     # res: (n, 4, embedding_dim)
        # Take only the last entry.
        res = res[:, -1, :]             # res: (n, 1, embedding_dim)
        res = res.view((n, -1))         # res: (n, embedding_dim)

        scores = res @ self.embedding.weight[:-4, :].t()    # scores: (n, num_embeddings)

        return scores
