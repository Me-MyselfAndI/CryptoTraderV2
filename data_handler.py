# Regression Example With Boston Dataset: Standardized and Wider
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import SGD, RMSprop, Nadam, Adadelta, Adagrad, Adam
import random, numpy as np, pandas
from sklearn.model_selection import train_test_split
import robin_stocks.robinhood.crypto as rs


data_frame = pandas.read_csv("data.csv", sep="\t")
data_frame = data_frame.sample(frac=1).reset_index(drop=True)
y = [data_frame["ask"], data_frame["bid"]]

data_list = []

data_np_array = data_frame.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(data_np_array, y[0], test_size=0.1)

best_error = 1000000

for attempt in range (15):
    neuron_amount = random.randint(5, round(x_train.shape[1]*2/3))
    model = Sequential([
        layers.Dense(neuron_amount, input_dim=x_train.shape[1], activation='relu'),
        layers.Dense(neuron_amount, input_dim=neuron_amount, activation='relu'),
        layers.Dense(1, input_dim=neuron_amount, activation='linear')
    ])


    model.compile(loss='mean_squared_logarithmic_error', optimizer=SGD(), metrics=['mse'])
    model.summary()


    history = model.fit(x_train, y_train, epochs=200, batch_size=1500, verbose=0)


    result = model.predict(x_test).tolist()
    result = [(result[i][0], y_test.tolist()[i], abs(result[i][0] - y_test.tolist()[i])/(y_test.tolist()[i]) if y_test.tolist()[i] != 0 else 0) for i in range(len(y_test))]

    errors = []
    for curr_res in result:
        errors.append(curr_res[-1])
        #to_print = "{%.7f}".format(curr_res[-1])
        print(curr_res[:-1], "\t\t", round(curr_res[-1], 6))

    errors.sort()
    median_error = errors[int(len(errors)/2)]
    print(len(errors))
    print(errors)
    print("Median error:", median_error)

    if best_error > median_error:
        best_model_num = attempt
        best_model = model
        best_error = median_error
        best_model_errors_list = errors

print("\n\n\n\n\n\nBEST MODEL:\t", best_model_num)
for error in best_model_errors_list:
    print(error)

median_error = best_model_errors_list[int(len(best_model_errors_list) / 2)]
print(best_model_errors_list)
print("Median error:", median_error)

print("Median error:", best_error)