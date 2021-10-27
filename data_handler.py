from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import SGD, Adagrad, Adam, Nadam, RMSprop
from sklearn.model_selection import train_test_split, StratifiedKFold as KFold
import random, pandas


class ModelBuilder:
    """
    Creates a model builder
    :param file: address file with the CSV data table
    :param dimension: column to use as Y ("ask" or "bid")
    :param test_ratio: percentage of input rows split out for testing (not used in model training)
    """
    def __init__(self, file, dimension, test_ratio=0.1):
        # Create a Pandas DataFrame to extract data from the CSV
        self.data_frame = pandas.read_csv(file, sep="\t")
        self.data_frame = self.data_frame.iloc[:-1, :]
        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)

        # Set the Y value to the "ask" and "bid" prices
        y = self.data_frame[dimension]
        # Remove Y from the data_frame
        self.data_frame = self.data_frame.drop("ask", axis=1).drop("bid", axis=1)
        # Divide the data into the train and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_frame.to_numpy(), y,
                                                                                test_size=test_ratio)

    """
    Compiles a particular number of models and selects the best 
    among them for prediction of the parameter

    :param num_folds: Number of times to retry compiling and fitting the model
    :rtype: Keras.Sequential
    :return: Model for prediction of the parameter
    """
    def make_model(self, num_folds=10):
        # the following code creates several models and finds the best to avoid sticking inside local minima
        best_error = float("inf")
        all_errors = []
        for attempt in range(num_folds):
            curr_x_train, curr_x_test, curr_y_train, curr_y_test = train_test_split(self.x_train, self.y_train,
                                                                                    test_size=0.2)
            # Create a model
            neurons_per_layer = random.randint(round(curr_x_train.shape[1] / 3), round(curr_x_train.shape[1]))
            model = Sequential([
                layers.Dense(neurons_per_layer, input_dim=curr_x_train.shape[1], activation='relu'),
                layers.Dense(neurons_per_layer, input_dim=neurons_per_layer, activation='relu'),
                layers.Dense(1, input_dim=neurons_per_layer, activation='linear')
            ])

            # compile and fit the model
            model.compile(loss='mean_squared_logarithmic_error', optimizer=RMSprop(learning_rate=0.002),
                          metrics=['mse'])
            model.fit(curr_x_train, curr_y_train, epochs=300,
                      batch_size=min(1500, int(curr_y_train.size / 2)), verbose=2)
            # compile and fit the model
            model.compile(loss='mean_squared_logarithmic_error', optimizer=RMSprop(learning_rate=0.001),
                          metrics=['mse'])
            model.fit(curr_x_train, curr_y_train, epochs=700,
                      batch_size=min(1500, int(curr_y_train.size / 2)), verbose=2)
            # compile and fit the model
            model.compile(loss='mean_squared_logarithmic_error', optimizer=RMSprop(learning_rate=0.0005),
                          metrics=['mse'])
            model.fit(curr_x_train, curr_y_train, epochs=1200,
                      batch_size=min(1500, int(curr_y_train.size / 2)), verbose=2)
            # compile and fit the model
            model.compile(loss='mean_squared_logarithmic_error', optimizer=SGD(learning_rate=0.1), metrics=['mse'])
            model.fit(curr_x_train, curr_y_train, epochs=1000,
                      batch_size=min(1500, int(curr_y_train.size / 2)), verbose=2)
            model.compile(loss='mean_squared_logarithmic_error', optimizer=SGD(learning_rate=0.01), metrics=['mse'])
            model.fit(curr_x_train, curr_y_train, epochs=3000,
                      batch_size=min(1500, int(curr_y_train.size / 2)), verbose=2)
            # compile and fit the model
            model.compile(loss='mean_squared_logarithmic_error', optimizer=SGD(learning_rate=0.005), metrics=['mse'])
            model.fit(curr_x_train, curr_y_train, epochs=3500,
                      batch_size=min(1500, int(curr_y_train.size / 2)), verbose=2)
            # compile and fit the model
            model.compile(loss='mean_squared_logarithmic_error', optimizer=SGD(learning_rate=0.001), metrics=['mse'])
            model.fit(curr_x_train, curr_y_train, epochs=4000,
                      batch_size=min(1500, int(curr_y_train.size / 2)), verbose=2)

            # Test the model on the TESTING data and record the result
            result = model.predict(curr_x_test).tolist()
            result = [(result[i][0], curr_y_test.tolist()[i],
                       abs(result[i][0] - curr_y_test.tolist()[i]) / (curr_y_test.tolist()[i])
                       if curr_y_test.tolist()[i] != 0 else 0) for i in range(len(curr_y_test))]

            # Find the average error
            errors = []
            for curr_res in result:
                errors.append(curr_res[-1])
                print(curr_res[:-1], "\t\t", round(curr_res[-1], 6))
            errors.sort()
            avg_error = sum(errors) / len(errors)

            # Print the errors and average
            print(errors)
            print(f"Average Error - Trial #{attempt + 1}:\t{round(avg_error, 8)}")
            # If the average error is better than the currently best average error, save this as a new best model
            if best_error > avg_error:
                best_model_num = "model num currently omitted"
                best_model = model
                best_error = avg_error
                best_model_errors_list = errors
                best_result = result
            all_errors.append(avg_error)

        # Print some stats about the best model
        print("\n\n\n\n\n\nBEST MODEL:\t", best_model_num)
        for error in best_model_errors_list:
            print(error)
        median_error = best_model_errors_list[int(round(len(best_model_errors_list)) / 2)]
        for result in best_result:
            print(result)
        print("Median error:", median_error)
        print("Best error:", best_error)
        print("FINISHED")

        return best_model


if __name__ == "__main__":
    ask_model_builder = ModelBuilder(r"data.csv", 'ask', test_ratio=0.15)
    ask_model = ask_model_builder.make_model(num_folds=15)

    x_test, y_test = ask_model_builder.x_test, ask_model_builder.y_test
    ask_results = ask_model.predict(x_test)
    ask_errors = []
    for expected, resulting in zip(y_test, ask_results):
        print(expected, resulting)
        ask_errors.append(abs((resulting - expected) / expected))

    avg_ask_error = sum(ask_errors) / len(ask_errors)
    median_ask_error = ask_errors[int(round(len(ask_errors) / 2))]

    bid_model_builder = ModelBuilder(r"data.csv", 'bid', test_ratio=0.15)
    bid_model = bid_model_builder.make_model(num_folds=15)

    x_test, y_test = bid_model_builder.x_test, bid_model_builder.y_test
    bid_results = bid_model.predict(x_test)
    bid_errors = []
    for expected, resulting in zip(y_test, bid_results):
        print(expected, resulting)
        bid_errors.append(abs((resulting - expected) / expected))

    avg_bid_error = sum(bid_errors) / len(bid_errors)
    median_bid_error = bid_errors[int(round(len(bid_errors) / 2))]

    print("Average ask error:\t", avg_ask_error)
    print("Median ask error:\t", median_ask_error)

    print("Average bid error:\t", avg_bid_error)
    print("Median bid error:\t", median_bid_error)