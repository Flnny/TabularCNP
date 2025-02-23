import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import Dnn, Linear, SimpleMLP


class DeterministicEncoder(nn.Module):
    def __init__(self, sizes, mask_sizes):
        super(DeterministicEncoder, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.linears.append(nn.Linear(sizes[i], sizes[i + 1]))

        self.out_linear = nn.Linear(sizes[i], 34)
        self.attention = nn.MultiheadAttention(embed_dim=34, num_heads=2, batch_first=True)
        self.mlp = SimpleMLP()
    def forward(self, context_x, context_y, mask_matrix, target_x):

        mask_matrix_expanded = mask_matrix.unsqueeze(-1)
        context_x = context_x * mask_matrix_expanded
        context_y = context_y * mask_matrix_expanded

        encoder_input = torch.cat((context_x, context_y), dim=-1)

        batch_size, set_size, filter_size = encoder_input.shape
        x = encoder_input.view(batch_size * set_size, -1)

        for i, linear in enumerate(self.linears[:-1]):
            x = torch.relu(linear(x))
        out = self.out_linear(x)

        target_x_3d = target_x.unsqueeze(1).expand(-1, 316, -1)
        out_3d = out.view(batch_size, set_size, -1)
        attention_out, _ = self.attention(context_x, target_x_3d, out_3d)

        attention_out = self.mlp(attention_out)
        representation = attention_out.mean(dim=1)

        return representation


class DeterministicDecoder(nn.Module):
    def __init__(self, sizes, representation_size, num_embeddings, embedding_dim, dense_features, sparse_features,
                 hidden_units, dnn_dropout=0.):

        super(DeterministicDecoder, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = dense_features, sparse_features
        self.cate_fea_size = len(num_embeddings)
        self.nume_fea_size = len(dense_features)

        if self.nume_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.nume_fea_size + representation_size, 1)
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in num_embeddings])

        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, embedding_dim) for voc_size in num_embeddings])

        hidden_units.insert(0,
                            len(self.dense_feature_cols) + len(self.sparse_feature_cols) * embedding_dim +
                            representation_size)
        self.dnn_network = Dnn(hidden_units)
        self.linear = Linear(len(self.dense_feature_cols) + representation_size)
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, representation, target_x):
        dense_input, sparse_inputs = target_x[:, :len(self.dense_feature_cols)], target_x[:,
                                                                                 len(self.dense_feature_cols):]

        dense_input = torch.cat([dense_input, representation], axis=-1)
        sparse_inputs = sparse_inputs.long()

        fm_1st_sparse_res = [emb(sparse_inputs[:, i].unsqueeze(1)).view(-1, 1)
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_fea_size]
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1, keepdim=True)  # [bs, 1]

        if dense_input is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(dense_input)
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res  # [bs, 1]

        fm_2nd_order_res = [emb(sparse_inputs[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)

        sum_embed = torch.sum(fm_2nd_concat_1d, 1)
        square_sum_embed = sum_embed * sum_embed
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d
        sum_square_embed = torch.sum(square_embed, 1)
        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5

        fm_2nd_part = torch.sum(sub, 1, keepdim=True)

        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)
        dnn_out = torch.cat([dnn_out, dense_input], axis=-1)

        dnn_out = self.dnn_network(dnn_out)

        dnn_out = self.final_linear(dnn_out)
        outputs = fm_1st_part + fm_2nd_part + dnn_out

        return outputs


class CDeepFM(nn.Module):
    def __init__(self, encoder_sizes, mask_sizes, decoder_sizes, representation_size, num_embeddings, embedding_dim, dense_features,
                 sparse_features, hidden_units=[256, 128, 64], dnn_dropout=0.):
        super(CDeepFM, self).__init__()
        self._encoder = DeterministicEncoder(encoder_sizes, mask_sizes)
        self._decoder = DeterministicDecoder(decoder_sizes, representation_size, num_embeddings, embedding_dim,
                                             dense_features, sparse_features, hidden_units, dnn_dropout)

    def forward(self, context_x, context_y, mask_matrix, target_x, target_y=None):
        representation = self._encoder(context_x, context_y, mask_matrix, target_x)
        outputs = self._decoder(representation, target_x)

        return outputs


