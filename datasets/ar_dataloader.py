import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.dataloader import TabularCNPDataset
from model.cDeepFM import CDeepFM


class ARTabularCNPCNPDataset(Dataset):
    def __init__(self, df, data, encoder_sizes, mask_sizes, decoder_sizes, representation_size, num_embeddings,
                 embedding_dim, dense_features, sparse_features, hidden_units, target):
        super().__init__()
        self.df = df
        self.data = data
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.target = target
        self.df.index = pd.to_datetime(self.df.index)
        self.mask_sizes = mask_sizes

        model_framework = CDeepFM(encoder_sizes, mask_sizes, decoder_sizes, representation_size, num_embeddings,
                             embedding_dim, dense_features, sparse_features, hidden_units).to('cpu')
        state_dict = torch.load('TabularCNPsh_cDeepFM.pth', map_location=torch.device('cpu'))
        model_framework.load_state_dict(state_dict)
        self.model = model_framework
        self.dataset = TabularCNPDataset(data, data, mask_sizes, sparse_features, dense_features, target)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        mask = np.zeros(self.mask_sizes[0])

        y_true = self.df.iloc[index][self.target]
        X_train = self.df.drop(self.target, axis=1).iloc[index]


        current_date = self.df.index[index]

        three_months_ago = current_date - pd.DateOffset(months=3)
        three_months_later = current_date + pd.DateOffset(months=3)

        prior_transactions = self.data[
            (self.data.index < current_date) &
            (self.data.index >= three_months_ago)
            ]

        nearby_found = 0

        context_x = np.zeros((self.mask_sizes[0], len(self.sparse_features) + len(self.dense_features)))
        context_y = np.zeros(self.mask_sizes[0])

        if not prior_transactions.empty:
            nearby_transactions = prior_transactions[
                ((prior_transactions['longitude'] - X_train['longitude']) ** 2 +
                 (prior_transactions['latitude'] - X_train['latitude']) ** 2) ** 0.5 < 0.01
                ]

            for _, transaction in nearby_transactions.iterrows():
                if nearby_found >= self.mask_sizes[0]:
                    break

                mask[nearby_found] = 1
                context_x[nearby_found] = transaction[self.dense_features + self.sparse_features].values
                context_y[nearby_found] = transaction[self.target].values
                nearby_found += 1

        df = self.data.reset_index()

        behind_transactions = df[
            (df['transaction_date'] > current_date) &
            (df['transaction_date'] <= three_months_later)
            ]

        if not behind_transactions.empty:
            nearby_transactions_b = behind_transactions[
                ((behind_transactions['longitude'] - X_train['longitude']) ** 2 +
                 (behind_transactions['latitude'] - X_train['latitude']) ** 2) ** 0.5 < 0.01
                ]
            for idx, transaction in nearby_transactions_b.iterrows():
                if nearby_found >= self.mask_sizes[0]:
                    break
                mask[nearby_found] = 1

                context_x[nearby_found] = transaction[self.dense_features + self.sparse_features].values
                context_x_ar, context_y_ar, mask_ar, X_train_ar, _ = self.dataset.__getitem__(idx)
                context_x_ar, context_y_ar, mask_tensor, X_train_ar = map(lambda t: t.cpu(),
                                                                          [context_x_ar, context_y_ar,
                                                                           mask_ar, X_train_ar])
                context_x_ar, context_y_ar, mask_tensor, X_train_ar = map(lambda x: x.unsqueeze(0),
                                                                          [context_x_ar, context_y_ar, mask_tensor,
                                                                           X_train_ar])
                context_y[nearby_found] = self.model(context_x_ar, context_y_ar, mask_tensor, X_train_ar)
                nearby_found += 1


        while nearby_found < self.mask_sizes[0]:
            context_x[nearby_found] = X_train[self.dense_features + self.sparse_features].values
            context_y[nearby_found] = 0
            nearby_found += 1

        context_x_tensor = torch.tensor(context_x, dtype=torch.float32)
        context_y_tensor = torch.tensor(context_y, dtype=torch.float32).view(-1, 1)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        y_true_tensor = torch.tensor(y_true.values, dtype=torch.float32)
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)

        if context_x_tensor.shape[0] != self.mask_sizes[0]:
            print('error')
        # print(nearby_found)
        return context_x_tensor, context_y_tensor, mask_tensor, X_train_tensor, y_true_tensor


if __name__ == '__main__':
    print('end')