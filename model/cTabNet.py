import torch
import torch.nn as nn
from model.base import Dnn, Linear
from torch.nn import Linear, BatchNorm1d, ReLU
from pytorch_tabnet.tab_network import TabNetEncoder, initialize_non_glu

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
    def __init__(
            self,
            input_dim,
            output_dim,
            representation_size,
            device,
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            epsilon=1e-15,
            virtual_batch_size=128,
            momentum=0.02,
            mask_type="sparsemax",
    ):
        super(DeterministicDecoder, self).__init__()
        self.input_dim = input_dim + representation_size
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = BatchNorm1d(self.input_dim, momentum=0.01)
        group_attention_matrix = torch.eye(self.input_dim).to(device)
        self.encoder = TabNetEncoder(
            input_dim=self.input_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=group_attention_matrix
        )

        if self.is_multi_task:
            self.multi_task_mappings = torch.nn.ModuleList()
            for task_dim in output_dim:
                task_mapping = Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
        else:
            self.final_mapping = Linear(n_d, output_dim, bias=False)
            initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, representation, x):
        res = 0
        x = torch.cat((x, representation), dim=-1)
        steps_output, M_loss = self.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        if self.is_multi_task:
            # Result will be in list format
            out = []
            for task_mapping in self.multi_task_mappings:
                out.append(task_mapping(res))
        else:
            out = self.final_mapping(res)
        return out


class CTabNet(nn.Module):
    def __init__(self, encoder_sizes, mask_sizes, decoder_sizes, representation_size, num_embeddings, embedding_dim, dense_features,
                 sparse_features, device, hidden_units=[256, 128, 64], dnn_dropout=0.):
        super(CTabNet, self).__init__()
        self._encoder = DeterministicEncoder(encoder_sizes, mask_sizes)
        self._decoder = DeterministicDecoder(input_dim=encoder_sizes[0]-1, output_dim=1, representation_size=representation_size,
                                             device=device)

    def forward(self, context_x, context_y, mask_matrix, target_x, target_y=None):
        representation = self._encoder(context_x, context_y, mask_matrix)
        outputs = self._decoder(representation, target_x)

        return outputs
