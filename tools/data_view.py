import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.stats import norm
import folium
from folium.plugins import HeatMap
from sklearn.preprocessing import MinMaxScaler

matplotlib.use('TkAgg')


def plot_combined_data(data_list, targets):
    fig_line, axs_line = plt.subplots(len(targets), len(data_list), figsize=(14, 8))
    fig_hist, axs_hist = plt.subplots(len(targets), len(data_list), figsize=(14, 8))

    for i, target in enumerate(targets):
        for j, data_name in enumerate(data_list):
            if data_name == 'xm':
                filename = f'{data_name}_pro.csv'
            else:
                filename = f'{data_name}_pro2.csv'
            data = pd.read_csv(filename, encoding='gbk')
            scaler = MinMaxScaler()
            data[target] = scaler.fit_transform(data[target].values.reshape(-1, 1))
            split_index = int(len(data) * 0.8)
            train_data = data.iloc[:split_index]
            test_data = data.iloc[split_index:]

            train = train_data[target].values.reshape((-1))
            test = test_data[target].values.reshape((-1))
            sns.histplot(train, bins=30, stat='density', ax=axs_hist[i, j], color='blue', alpha=0.5, label='Train&Valid')
            sns.histplot(test, bins=30, stat='density', ax=axs_hist[i, j], color='orange', alpha=0.5, label='Test')

            mu_train, std_train = norm.fit(train)
            mu_test, std_test = norm.fit(test)

            x = np.linspace(min(train.min(), test.min()),
                            max(train.max(), test.max()), 100)
            p_train = norm.pdf(x, mu_train, std_train)
            p_test = norm.pdf(x, mu_test, std_test)

            axs_hist[i, j].plot(x, p_train, 'k', linewidth=2, label='Train Normal Distribution')
            axs_hist[i, j].plot(x, p_test, 'r', linewidth=2, label='Test Normal Distribution')

            axs_hist[i, j].set_xlabel('Unit price')
            axs_hist[i, j].set_ylabel('Frequency')
            axs_hist[i, j].legend()
    plt.tight_layout()
    plt.savefig('data_price.png')
    plt.show()




def plot_heatmaps(data_list, targets, center_list):
    for i, target in enumerate(targets):
        for j, data_name in enumerate(data_list):
            filename = f'E:\\postgraduate\\cWDL\\datasets\\house\\{data_name}_pro.csv'
            data = pd.read_csv(filename, encoding='gbk')

            m = folium.Map(location=center_list[j], zoom_start=5, tiles='CartoDB dark_matter')

            heat_data = [[row['latitude'], row['longitude'], row[target]] for index, row in data.iterrows()]

            HeatMap(
                heat_data,
                radius=15,
                blur=10,
                max_zoom=1,
                min_opacity=0.3,
                gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
            ).add_to(m)

            html_name = data_name + '_heatmap.html'
            m.save(html_name)



data_name_list = ['xiameng', 'nanjing', 'shanghai']
data_list = ['xm', 'nj', 'sh']
# center_list = [[24.4798, 118.0894], [30.5728, 104.0668], [32.0617, 118.7778]]
# data_name_list = ['shanghai']
# data_list = ['sh']
# center_list = [[31.2304, 121.4737]]
targets = ['avg_transaction', 'transaction_cycle']

# plot_heatmaps(data_list, targets, center_list)

plot_combined_data(data_list, targets)