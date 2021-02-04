import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# import the training data
dataset_train = pd.read_csv('recurrent_neural_network/Google_Stock_Price_Train.csv')
training_set = dataset_train[['Open']].to_numpy()

# feature scaling
scaler = MinMaxScaler()
training_set_scaled = scaler.fit_transform(training_set)

# creating a data structure with 60 timesteps and 1 output, 60 was selected to get the best accuracy
X_train = list()
y_train = list()

for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# reshape to fit RNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# build RNN
regressor = Sequential()

# input layer
neurons = 128
regressor.add(LSTM(units=neurons, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))

# hidden layers
regressor.add(LSTM(units=neurons, return_sequences=True))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=neurons, return_sequences=True))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=neurons))
regressor.add(Dropout(rate=0.2))

# output layer
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(X_train, y_train, epochs=200, batch_size=32)

# make predictions
dataset_test = pd.read_csv('recurrent_neural_network/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test[['Open']].to_numpy()

inputs = X_train[-60:]

predictions = list()

for _ in range(len(real_stock_price)):
    prediction = regressor.predict(inputs)
    predictions.append(prediction[-1][0])
    inputs = np.delete(inputs, 0, axis=0)
    reshaped_prediction = np.reshape(prediction, (1, prediction.shape[0], 1))
    inputs = np.append(inputs, reshaped_prediction, axis=0)

predictions = np.array(predictions)
predictions = np.reshape(predictions, (predictions.shape[0], 1))

predicted_stock_price = scaler.inverse_transform(predictions)

plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

np.savetxt('recurrent_neural_network/predicted_stock_price_future.csv', predicted_stock_price.T, delimiter=',')
