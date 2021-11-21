import pyotp
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import SGD, Adagrad, Adam, Nadam, RMSprop
from sklearn.model_selection import train_test_split, StratifiedKFold as KFold
import random, pandas

import numpy
import robin_stocks.robinhood as rs
from data_digger import DataDigger


class ModelBuilder:
    """
    Creates a model builder
    :param file: address file with the CSV data table
    :param dimension: column to use as Y ("ask" or "bid")
    :param test_ratio: percentage of input rows split out for testing (not used in model training)
    """
    def __init__(self, file, dimension, test_ratio=0.1):
        # Create a Pandas DataFrame to extract data from the CSV
        self.raw_data_frame = pandas.read_csv(file, sep="\t")
        self.raw_data_frame = self.raw_data_frame.iloc[20:-1, :]
        self.raw_data_frame = self.raw_data_frame.sample(frac=1).reset_index(drop=True)

        self.data_frame = self.reshape_table()

        # Set the Y value to the "ask" and "bid" prices
        y = self.data_frame[dimension]
        # Remove Y from the data_frame
        self.data_frame = self.data_frame.drop("ask", axis=1).drop("bid", axis=1)
        # Divide the data into the train and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_frame.to_numpy(), y,
                                                                                test_size=test_ratio)
    def reshape_row (self, data_row):
        data_row = list(data_row)
        raw_width = len(data_row)
        new_row_values = []
        # Change the ask prices
        for col_num in range(int(raw_width / 2 - 2)):
            new_row_values.append(data_row[col_num] - data_row[col_num + 1])
        # Change the bid prices
        for col_num in range(int(raw_width / 2) - 1, raw_width - 3):
            new_row_values.append(data_row[col_num] - data_row[col_num + 1])

        new_row_values.extend(data_row[-2:])
        return new_row_values


    def reshape_table (self):
        raw_height, raw_width = self.raw_data_frame.shape

        data_frame = pandas.DataFrame(columns=['ask'] + [('a' + str(i)) for i in range(int(raw_width / 2) - 3)] +
                                                   ['bid'] + [('b' + str(i)) for i in range(int(raw_width / 2) - 3)] +
                                                   ['ask_pade', 'bid_pade'])
        # Extract delta_price (price change) from prices
        for row_num in range(raw_height):
            curr_row = self.raw_data_frame.iloc[row_num]
            column_names = list(self.raw_data_frame.columns)

            new_row_values = []
            # Change the ask prices
            for col_num in range(int(raw_width / 2 - 2)):
                new_row_values.append(curr_row.iloc[col_num] - curr_row.iloc[col_num + 1])
            # Change the bid prices
            for col_num in range(int(raw_width / 2) - 1, raw_width - 3):
                new_row_values.append(curr_row.iloc[col_num] - curr_row.iloc[col_num + 1])
            new_row_values.extend(curr_row[-2:])
            data_frame.loc[row_num] = new_row_values

        return data_frame

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
        attempt = 0
        while attempt < num_folds:
            curr_x_train, curr_x_test, curr_y_train, curr_y_test = train_test_split(self.x_train, self.y_train,
                                                                                    test_size=0.2)
            # Create a model
            neurons_per_layer = random.randint(round(curr_x_train.shape[1] / 3), round(curr_x_train.shape[1]))
            model = Sequential([
                layers.Dense(neurons_per_layer, input_dim=curr_x_train.shape[1], activation='relu'),
                layers.Dense(neurons_per_layer, input_dim=neurons_per_layer, activation='relu'),
#                layers.Dense(neurons_per_layer, input_dim=neurons_per_layer, activation='relu'),
                layers.Dense(1, input_dim=neurons_per_layer, activation='linear')
            ])

            # compile and fit the model
            model.compile(loss='huber', optimizer=RMSprop(learning_rate=0.005),
                          metrics=['mae'])
            history = model.fit(curr_x_train, curr_y_train, epochs=500,
                                batch_size=min(1500, int(curr_y_train.size / 2)), verbose=0)
            # If average is more than a threshold
            #if sum(history.history['loss']) > 35 * len(history.history['loss']):
            #    print("\n\n\n\n\n\n\n\nStuck in a local min\n\n\n\n\n\n\n\n")
            #    continue
            print(end='|')

            # compile and fit the model
            model.compile(loss='huber', optimizer=RMSprop(learning_rate=0.002),
                          metrics=['mae'])
            history = model.fit(curr_x_train, curr_y_train, epochs=100,
                                batch_size=min(1500, int(curr_y_train.size / 2)), verbose=0)
            print(end='->')

            # compile and fit the model
            model.compile(loss='huber', optimizer=SGD(learning_rate=0.00002),
                          metrics=['mae'])
            model.fit(curr_x_train, curr_y_train, epochs=1000,
                      batch_size=min(1500, int(curr_y_train.size / 2)), verbose=0)
            print(end='|')

            # Run the model on the testing data and record the result
            result = model.predict(curr_x_test).tolist()
            result = [(result[i][0], curr_y_test.tolist()[i],
                       abs(result[i][0] - curr_y_test.tolist()[i]) / (curr_y_test.tolist()[i])
                       if curr_y_test.tolist()[i] != 0 else 0) for i in range(len(curr_y_test))]

            # Find the average error
            errors = []
            for curr_res in result:
                errors.append(abs(curr_res[-1]))
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
            attempt += 1

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
    ask_model_builder = ModelBuilder(r"coefs.csv", 'ask', test_ratio=0.15)
    ask_model = ask_model_builder.make_model(num_folds=5)

    x_test, y_test = ask_model_builder.x_test, ask_model_builder.y_test
    ask_results = ask_model.predict(x_test)
    ask_errors = []
    for expected, resulting in zip(y_test, ask_results):
        print(expected, resulting)
        ask_errors.append(abs((resulting - expected) / expected))

    avg_ask_error = sum(ask_errors) / len(ask_errors)
    median_ask_error = ask_errors[int(round(len(ask_errors) / 2))]

    bid_model_builder = ModelBuilder(r"coefs.csv", 'bid', test_ratio=0.15)
    bid_model = bid_model_builder.make_model(num_folds=5)

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


    totp = pyotp.TOTP("YHYPKMLAURGON3I2").now()
    rs.login('p.grigorii01@gmail.com', 'Alexpodoksik#66', mfa_code=totp)
    data_digger = DataDigger(r"temp_coefs.csv", asset_code='BTC', prediction_delay=100)
    data_digger.fill_data_table(100)

    i = 0
    while True:
        i += 1
        try:
            print(ask_model.predict(numpy.array(ask_model_builder.reshape_row(data_digger.get_data_row(for_training=False))).reshape(1, 18)))
        except Exception as e:
            print(e)
            print('waited', i, 'times')