import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import Dnn, Linear, WideDeep


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
                 hidden_units, dnn_dropout=0.):
        super(DeterministicDecoder, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = dense_features, sparse_features

        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=num_embeddings[i], embedding_dim=embedding_dim)
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        hidden_units.insert(0,
                            len(self.dense_feature_cols) + len(self.sparse_feature_cols) * embedding_dim +
                            representation_size)
        self.dnn_network = Dnn(hidden_units)
        self.linear = Linear(len(self.dense_feature_cols) + representation_size)
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, representation, target_x):
        dense_input, sparse_inputs = target_x[:, :len(self.dense_feature_cols)], target_x[:,
                                                                                 len(self.dense_feature_cols):]
        sparse_inputs = sparse_inputs.long()

        sparse_embeds = []
        for i in range(sparse_inputs.shape[1]):
            try:
                embed_key = 'embed_' + str(i)
                if embed_key in self.embed_layers:
                    embed_layer = self.embed_layers[embed_key]
                    sparse_embeds.append(embed_layer(sparse_inputs[:, i]))
                else:
                    print(f"Warning: {embed_key} not found in embed_layers.")
            except IndexError as e:
                print(f"IndexError at column {i}: {e}")

        sparse_embeds = torch.cat(sparse_embeds, axis=-1)

        dense_input = torch.cat([dense_input, representation], axis=-1)

        dnn_input = torch.cat([sparse_embeds, dense_input], axis=-1)

        # Wide
        wide_out = self.linear(dense_input)

        # Deep
        deep_out = self.dnn_network(dnn_input)
        deep_out = self.final_linear(deep_out)
        outputs = 0.5 * (wide_out + deep_out)
        return outputs


class CWDL(nn.Module):
    def __init__(self, encoder_sizes, mask_sizes, decoder_sizes, representation_size, num_embeddings, embedding_dim, dense_features,
                 sparse_features, hidden_units, dnn_dropout=0.):
        super(CWDL, self).__init__()
        self._encoder = DeterministicEncoder(encoder_sizes, mask_sizes)
        self._decoder = DeterministicDecoder(decoder_sizes, representation_size, num_embeddings, embedding_dim,
                                             dense_features, sparse_features, hidden_units, dnn_dropout)

    def forward(self, context_x, context_y, mask_matrix, target_x, target_y=None):
        representation = self._encoder(context_x, context_y, mask_matrix)
        outputs = self._decoder(representation, target_x)

        return outputs


