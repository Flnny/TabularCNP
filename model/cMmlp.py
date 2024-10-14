import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import Dnn, Linear


class DeterministicEncoder(nn.Module):
    def __init__(self, sizes, mask_sizes):
        super(DeterministicEncoder, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.linears.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, context_x, context_y, mask_matrix):

        mask_matrix_expanded = mask_matrix.unsqueeze(-1)
        context_x = context_x*mask_matrix_expanded
        context_y = context_y*mask_matrix_expanded
        encoder_input = torch.cat((context_x, context_y), dim=-1)
        batch_size, set_size, filter_size = encoder_input.shape
        x = encoder_input.view(batch_size * set_size, -1)
        for i, linear in enumerate(self.linears[:-1]):
            x = torch.relu(linear(x))
        x = self.linears[-1](x)
        x = x.view(batch_size, set_size, -1)
        representation = x.mean(dim=1)
        return representation


class DeterministicDecoder(nn.Module):
    def __init__(self, sizes, representation_size, num_embeddings, embedding_dim, dense_features, sparse_features,
                 hidden_units, dropout=0.0):
        super(DeterministicDecoder, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.linears.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, representation, target_x):

        x = torch.cat((representation, target_x), dim=-1)
        for i, linear in enumerate(self.linears[:-1]):
            x = torch.relu(linear(x))
        output = self.linears[-1](x)

        return output


class CMLP(nn.Module):
    def __init__(self, encoder_sizes, mask_sizes, decoder_sizes, representation_size, num_embeddings, embedding_dim, dense_features,
                 sparse_features, hidden_units=[256, 128, 64], dnn_dropout=0.):
        super(CMLP, self).__init__()
        self._encoder = DeterministicEncoder(encoder_sizes, mask_sizes)
        self._decoder = DeterministicDecoder(decoder_sizes, representation_size, num_embeddings, embedding_dim,
                                             dense_features, sparse_features, hidden_units, dnn_dropout)

    def forward(self, context_x, context_y, mask_matrix, target_x, target_y=None):
        representation = self._encoder(context_x, context_y, mask_matrix)
        outputs = self._decoder(representation, target_x)

        return outputs