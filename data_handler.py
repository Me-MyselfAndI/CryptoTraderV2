# Regression Example With Boston Dataset: Standardized and Wider
from keras import layers, optimizers, Sequential
import random, numpy as np, pandas
from sklearn.model_selection import train_test_split
import robin_stocks.robinhood.crypto as rs


data_frame = pandas.read_csv("data.csv", sep="\t")
data_frame = data_frame.sample(frac=1).reset_index(drop=True)
y = [data_frame["ask"], data_frame["bid"]]

data_list = []

data_np_array = data_frame.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(data_np_array, y[0], test_size=0.1)

model = Sequential([
    layers.Dense(100, input_dim=x_train.shape[1], activation='relu'),
    layers.Dense(200, input_dim=100, activation='relu'),
    layers.Dense(400, input_dim=200, activation='relu'),
    layers.Dense(1, input_dim=400, activation='linear')
])

#optimizers.Adam(lr=0.0001, beta_1=0.9, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
model.summary()

history = model.fit(x_train, y_train, epochs=4000, batch_size=500)
history = model.fit(x_train, y_train, epochs=5000, batch_size=200)
history = model.fit(x_train, y_train, epochs=4000, batch_size=100)
history = model.fit(x_train, y_train, epochs=3000, batch_size=50)

result = model.predict(x_test).tolist()
result = [(result[i][0], y_test.tolist()[i], abs(result[i][0] - y_test.tolist()[i])/(y_test.tolist()[i]) if y_test.tolist()[i] != 0 else 0) for i in range(len(y_test))]

errors = []
for curr_res in result:
    errors.append(curr_res[-1])
    #to_print = "{%.7f}".format(curr_res[-1])
    print(curr_res[:-1], "\t\t", round(curr_res[-1], 6))

errors.sort()
print(len(errors))
print(errors)
print("Median error:", errors[int(len(errors)/2)])