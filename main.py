import datetime
import logging

import numpy as np
import pandas as pd
import torch
from deepctr.feature_column import SparseFeat, DenseFeat
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, auc, roc_auc_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import optim, nn
from torch.optim import optimizer
from torch.utils.data import DataLoader
import matplotlib

from datasets.dataset import target, features, dense_features, sparse_features

matplotlib.use('TkAgg')
from datasets.dataloader import TabularCNPDataset
from model.cWDL import CWDL


df = pd.read_csv('xm_pro.csv')
#data processing
data = df.copy()
data = data[target + features]

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
data[dense_features] = scaler_x.fit_transform(data[dense_features])
data[target] = scaler_y.fit_transform(data[target])
split_index = int(len(data) * 0.8)
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

#model parameter
batch_size = 128
d_x, d_in, d_out = 1, 35, 2
representation_size = 10
encoder_sizes = [d_in, 128, 128, 128, representation_size]
decoder_sizes = [representation_size + d_x, 128, 128, 2]
mask_sizes = [316, batch_size]
hidden_units = [256, 128, 64]
num_embeddings = [max(data[feat])+1 for i, feat in enumerate(sparse_features)]
embedding_dim = 4

#Training Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CWDL(encoder_sizes, mask_sizes, decoder_sizes, representation_size, num_embeddings, embedding_dim, dense_features, sparse_features, hidden_units).to(device)
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
metric_func = mean_squared_error
metric_name = 'mse'
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#dataloader
test_dataset = TabularCNPDataset(test_data, data, mask_sizes, sparse_features, dense_features, target)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train, valid = train_test_split(train_data, test_size=0.2, random_state=2024)
train_dataset = TabularCNPDataset(train, data, mask_sizes, sparse_features, dense_features, target)
valid_dataset = TabularCNPDataset(valid, data, mask_sizes, sparse_features, dense_features, target)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# Early stop
num_epochs = 500
patience = 5
best_val_loss = float('inf')
patience_counter = 0
dfhistory = pd.DataFrame(columns=['epoch', 'loss', metric_name, 'val_loss', 'val_' + metric_name])

print('start_training...')
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('========' * 8 + '%s' % nowtime)

for epoch in range(num_epochs):
    model.train()
    loss_sum = 0.0
    metric_sum = 0.0

    for step, (context_x, context_y, mask_matrix, X_train, y_true) in enumerate(train_dataloader):
        context_x, context_y, mask_matrix, X_train, y_true = context_x.to(device), context_y.to(device), mask_matrix.to(device), X_train.to(
            device), y_true.to(device)  # GPU

        optimizer.zero_grad()
        outputs = model(context_x, context_y, mask_matrix, X_train)
        loss = loss_func(outputs, y_true)

        metric = metric_func(y_true.cpu().numpy(), outputs.detach().cpu().numpy())
        print(f"train_mse:{metric}")
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        metric_sum += metric

    model.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0

    for val_step, (context_x, context_y, mask_matrix, X_train, y_true) in enumerate(valid_dataloader):
        context_x, context_y, mask_matrix, X_train, y_true = context_x.to(device), context_y.to(device), mask_matrix.to(
            device), X_train.to(device), y_true.to(device) 

        with torch.no_grad():
            outputs = model(context_x, context_y, mask_matrix, X_train)
            val_loss = loss_func(outputs, y_true)
            val_metric = metric_func(y_true.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            print(f"valid_mse:{val_metric}")

        val_loss_sum += val_loss.item()
        val_metric_sum += val_metric

        # record
    info = (epoch, loss_sum / len(train_dataloader), metric_sum / len(train_dataloader), val_loss_sum / len(valid_dataloader),
    val_metric_sum / len(valid_dataloader))
    dfhistory.loc[epoch] = info


    print(("\nEPOCH=%d, loss=%.3f, " + metric_name + " = %.3f, val_loss=%.3f, " + "val_" + metric_name + " = %.3f") % info)
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('\n' + '==========' * 8 + '%s' % nowtime)

    # Early stop
    if val_loss_sum / len(valid_dataloader) < best_val_loss:
        best_val_loss = val_loss_sum / len(valid_dataloader)
        patience_counter = 0  
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stop trigger, stop training.")
            break
    scheduler.step()

print('Finished Training')


print('start_testing.........')
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('========' * 8 + '%s' % nowtime)
model.eval()
test_loss_sum = 0.0
test_metric = 0.0
test_metric_sum = 0.0
test_step = 1
mae_sum = 0.0

outputs_list = []
y_true_list = []

for test_step, (context_x, context_y, mask_matrix, X_train, y_true) in enumerate(test_dataloader, 1):
    with torch.no_grad():
        context_x, context_y, mask_matrix, X_train, y_true = context_x.to(device), context_y.to(device), mask_matrix.to(
            device), X_train.to(device), y_true.to(device)
        outputs = model(context_x, context_y, mask_matrix, X_train)

        outputs_list.append(outputs.cpu().detach().numpy())
        y_true_list.append(y_true.cpu().detach().numpy())


all_outputs = np.concatenate(outputs_list, axis=0)
all_y_true = np.concatenate(y_true_list, axis=0)
all_outputs = scaler_y.inverse_transform(all_outputs)
all_y_true = scaler_y.inverse_transform(all_y_true)

mae = mean_absolute_error(all_y_true, all_outputs)
mape = mean_absolute_percentage_error(all_y_true, all_outputs) * 100
try:
    test_metric = metric_func(all_y_true, all_outputs)
except ValueError:
    test_metric = None
rmse = np.sqrt(test_metric)

print(f'mae:{mae}')
print(f'rmse:{rmse}')
print(f'mape:{mape}')

nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('\n' + '==========' * 8 + '%s' % nowtime)
print('Finished Testing')


