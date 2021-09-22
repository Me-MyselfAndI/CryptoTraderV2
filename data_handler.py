from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import SGD, RMSprop, Nadam, Adadelta, Adagrad, Adam
import random, numpy as np, pandas
from sklearn.model_selection import train_test_split
import robin_stocks.robinhood.crypto as rs

# Create a Pandas DataFrame to extract data from the CSV
data_frame = pandas.read_csv("data.csv", sep="\t")
data_frame = data_frame.sample(frac=1).reset_index(drop=True)

# Set the Y value to the "ask" and "bid" prices
y = [data_frame["ask"], data_frame["bid"]]
# Remove Y from the data_frame
data_frame = data_frame.drop("ask", axis=1).drop("bid", axis=1)

# the following code creates several models and finds the best to avoid sticking inside local minima
best_error = float("inf")
for attempt in range(25):
    # Divide the data into the train and test data
    x_train, x_test, y_train, y_test = train_test_split(data_frame.to_numpy(), y[0], test_size=0.1)

    # Create a model
    neurons_per_layer = random.randint(5, round(x_train.shape[1]*2/3))
    model = Sequential([
        layers.Dense(neurons_per_layer, input_dim=x_train.shape[1], activation='relu'),
        layers.Dense(neurons_per_layer, input_dim=neurons_per_layer, activation='relu'),
        layers.Dense(1, input_dim=neurons_per_layer, activation='linear')
    ])

    # compile and fit the model
    model.compile(loss='mean_squared_logarithmic_error', optimizer=SGD(), metrics=['mse'])
    history = model.fit(x_train, y_train, epochs=200, batch_size=1500, verbose=0)

    # Test the model on the TESTING data and record the resultd
    result = model.predict(x_test).tolist()
    result = [(result[i][0], y_test.tolist()[i], abs(result[i][0] - y_test.tolist()[i])/(y_test.tolist()[i]) if y_test.tolist()[i] != 0 else 0) for i in range(len(y_test))]

    # Find the median error
    errors = []
    for curr_res in result:
        errors.append(curr_res[-1])
        #to_print = "{%.7f}".format(curr_res[-1])
        print(curr_res[:-1], "\t\t", round(curr_res[-1], 6))
    errors.sort()
    median_error = errors[int(len(errors)/2)]

    # Print the errors and median
    print(errors)
    print("Median error:", median_error)

    # If the median error is better than the currently bet median error, save this as a new best model
    if best_error > median_error:
        best_model_num = attempt
        best_model = model
        best_error = median_error
        best_model_errors_list = errors
        best_result = result

# Print some stats about the best model
print("\n\n\n\n\n\nBEST MODEL:\t", best_model_num)
for error in best_model_errors_list:
    print(error)
median_error = best_model_errors_list[int(len(best_model_errors_list) / 2)]
for result in best_result:
    print(result)
print("Median error:", best_error)