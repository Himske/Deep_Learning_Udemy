import numpy as np
import pandas as pd
from minisom import MiniSom
from matplotlib.pylab import bone, pcolor, colorbar, plot, show
from sklearn.preprocessing import MinMaxScaler

# import and split dataset
dataset = pd.read_csv('self_organizing_map/Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# train the SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1, learning_rate=0.5)  # input len is the number of columns in the data
som.random_weights_init(X)
som.train_random(data=X, num_iteration=1000)

# visualizing the result
bone()
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5,
         markers[y[i]], markeredgecolor=colors[y[i]], markerfacecolor='None', markersize=10, markeredgewidth=2)

show()

# get the outliers and find the frauds
outliers = list()

distance_map = som.distance_map().T

for i in range(10):
    for j in range(10):
        if distance_map[i][j] >= 0.95:  # threshold for detecting outliers
            outliers.append((i, j))

mappings = som.win_map(X)

outlier_maps = list()

for k in outliers:
    if len(mappings[k]) > 0:
        outlier_maps.append(mappings[k])

if outlier_maps:
    frauds = np.concatenate(outlier_maps, axis=0)
    frauds = scaler.inverse_transform(frauds)

    np.savetxt(fname='self_organizing_map/Potential_Frauds.csv', X=frauds, fmt='%.3f', delimiter=',',
               header='CustomerID,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14')
