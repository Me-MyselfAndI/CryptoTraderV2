import math, statistics

import pyotp, numpy, random, pandas
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adagrad, Adam, Nadam, RMSprop

from sklearn.model_selection import train_test_split, StratifiedKFold as KFold
from sklearn.preprocessing import MinMaxScaler

import robin_stocks.robinhood as rs
from data_digger import DataDigger


def evaluate_prediction_quality(prediction_row, expected_row):
    max_prediction, max_prediction_index, max_expected, max_expected_index = 0, -1, 0, -1
    for i, prediction, expected in zip(range(len(prediction_row)), prediction_row, expected_row):
        if prediction > max_prediction:
            max_prediction = prediction
            max_prediction_index = i
        if expected > max_expected:
            max_expected = expected
            max_expected_index = i
    return max_expected_index == max_prediction_index

class ModelBuilder:
    """
    Creates a model builder
    :param file: address file with the CSV data table
    :param dimension: column to use as Y ("ask" or "bid")
    :param test_ratio: percentage of input rows split out for testing (not used in model training)
    """
    def __init__(self, file, test_ratio=0.3, verbose=0):
        # Create a Pandas DataFrame to extract data from the CSV
        self.raw_data_frame = pandas.read_csv(file, sep=",", index_col=False)
        self.raw_data_frame = self.raw_data_frame.iloc[:-1, :]
        self.raw_data_frame = self.raw_data_frame.sample(frac=1).reset_index(drop=True)

        self.data_frame = self.reshape_table()
        num_buy, num_sell, num_hold = 0, 0, 0
        for i in range(self.data_frame.shape[0]):
            if self.data_frame.iloc[i, 0] == 1:
                num_buy += 1
            elif self.data_frame.iloc[i, 1] == 1:
                num_sell += 1
            else:
                num_hold += 1

        min_x_trials_amount = statistics.median([num_hold, num_buy, num_sell])
        num_buy = min_x_trials_amount
        num_sell = min_x_trials_amount
        num_hold = min_x_trials_amount
        temp_data_frame = pandas.DataFrame(columns=self.data_frame.columns)
        for i in range(self.data_frame.shape[0]):
            if self.data_frame.iloc[i, 0] == 1 and num_buy > 0:
                num_buy -= 1
                temp_data_frame = temp_data_frame.append(self.data_frame.iloc[i])
            elif self.data_frame.iloc[i, 1] == 1 and num_sell > 0:
                num_sell -= 1
                temp_data_frame = temp_data_frame.append(self.data_frame.iloc[i])
            elif self.data_frame.iloc[i, 2] == 1 and num_hold > 0:
                num_hold -= 1
                temp_data_frame = temp_data_frame.append(self.data_frame.iloc[i])
            elif num_buy <= 0 and num_sell <= 0 and num_hold <= 0:
                break

        self.data_frame = temp_data_frame

        # Set the Y value to the "ask" and "bid" prices
        y = [self.data_frame['buy'], self.data_frame['sell'], self.data_frame['hold']]
        # Remove Y from the data_frame
        self.data_frame = self.data_frame.drop("buy", axis=1).drop("sell", axis=1).drop("hold", axis=1)

        raw_height, raw_width = self.data_frame.shape
        self.min_x, self.max_x = [math.inf for i in range (raw_width)], [-math.inf for i in range (raw_width)]
        for i in range(raw_height):
            for j in range(raw_width):
                if self.data_frame.iloc[i, j] > self.max_x[j]:
                    self.max_x[j] = self.data_frame.iloc[i, j]
                if self.data_frame.iloc[i, j] < self.min_x[j]:
                    self.min_x[j] = self.data_frame.iloc[i, j]

        temp_data_frame = self.data_frame
        for row_num in range(raw_height):
            curr_row = self.data_frame.iloc[row_num]
            new_row_values = []

            for col_num in range(raw_width):
                new_row_values.append((curr_row.iloc[col_num] - self.min_x[col_num])/(self.max_x[col_num] - self.min_x[col_num]))
            temp_data_frame.iloc[row_num] = new_row_values
        self.data_frame = temp_data_frame

        # Divide the data into the train and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_frame.to_numpy(),
                                                                                pandas.DataFrame(y).T, test_size=test_ratio)

        self.verbose = verbose

        #self.x_train = numpy.reshape(self.x_train, (self.x_train.shape[0], 1, self.x_train.shape[1]))
        #self.x_test = numpy.reshape(self.x_test, (self.x_test.shape[0], 1, self.x_test.shape[1]))

    def reshape_row(self, data_row):
        data_row = list(data_row)
        raw_width = len(data_row)
        new_row_values = []

        for col_num in range(raw_width):
            new_row_values.append(data_row[col_num])
        # new_row_values.extend(curr_row[-2:])
        for col_num in range(raw_width):
            new_row_values[col_num] = (new_row_values[col_num] - self.min_x[col_num]) / (self.max_x[col_num] - self.min_x[col_num])
        return pandas.DataFrame(new_row_values).T

    def reshape_table(self):
        raw_height, raw_width = self.raw_data_frame.shape
        data_frame = pandas.DataFrame(columns=['buy', 'sell', 'hold'] +
                                              [('a' + str(i)) for i in range(int(raw_width / 2) - 2)] +
                                              [('b' + str(i)) for i in range(int(raw_width / 2) - 2)] +
                                              ['ask_pade', 'bid_pade'])

        # Extract delta_price (price change) from prices
        for row_num in range(raw_height):
            curr_row = self.raw_data_frame.iloc[row_num]
            new_row_values = []

            delta_price = {'ask': curr_row.iloc[0] - curr_row.iloc[int(raw_width / 2) - 2],
                           'bid': curr_row.iloc[int(raw_width / 2) - 1] - curr_row.iloc[raw_width - 3]}
            delta_price['overall'] = (delta_price['ask'] + delta_price['bid'])/2

            new_row_values.append(1 if delta_price['overall'] > 5.0 else 0)   # whether to buy
            new_row_values.append(1 if delta_price['overall'] < -5.0 else 0)   # whether to sell
            new_row_values.append(1 if -5.0 <= delta_price['overall'] <= 5.0 else 0)    # whether to hold
            # Change the ask prices
            for col_num in range(1, int(raw_width / 2) - 1):
                new_row_values.append(curr_row.iloc[col_num])
            # Change the bid prices
            for col_num in range(int(raw_width / 2), raw_width):
                new_row_values.append(curr_row.iloc[col_num])
            #new_row_values.extend(curr_row[-2:])
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
            curr_x_train, curr_x_test, curr_y_train, curr_y_test = train_test_split(self.x_train, self.y_train, test_size=0.2)
            # Create a model
            neurons_per_layer = random.randint(round(curr_x_train.shape[1] / 3), round(curr_x_train.shape[1]))
            model = Sequential([
                Dense(500, input_shape=curr_x_train.shape, activation='relu'),
                Dropout(0.1),
                Dense(500, input_dim=500, activation='relu'),
                Dropout(0.1),
                Dense(500, input_dim=500, activation='relu'),
                Dense(3, input_dim=500, activation='softmax')
                #LSTM(500, activation='relu', return_sequences=True),
                #Dropout(0.05),
                #LSTM(500, activation='relu'),
                #Dense(1, input_dim=500, activation='softmax')
            ])

            # compile and fit the model
            if self.verbose >= 1:
                print(f"\nTraining the model: Fold {attempt+1}/{num_folds}")
            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=random.randint(1, 10) * 1e-4,
                                                                             decay=random.randint(1, 10) * 1e-6),
                          metrics=['accuracy'])
            model.fit(curr_x_train, curr_y_train, epochs=random.randint(20, 40) * 10,
                      batch_size=1000, verbose=self.verbose)

            if self.verbose >= 1:
                print(f"Finsihed Training")


            # Run the model on the testing data and record the result
            result = model.predict(curr_x_test)

            result = [(result[i], curr_y_test.iloc[i], evaluate_prediction_quality(result[i], curr_y_test.iloc[i])) for i in range(len(result))]

            # Find the percentage of correct
            errors = []
            for curr_res in result:
                errors.append(1 if curr_res[-1] else 0)
                print(curr_res[:-1], "\t\t", curr_res[-1])
            #errors.sort()
            avg_error = 1 - sum(errors) / len(errors)

            if self.verbose == 2:
                # Print the errors and average
                print(errors)
            if self.verbose >= 1:
                print(f"Average Error - Trial #{attempt + 1}:\t{round(avg_error, 8)}")
            # If the average error is better than the currently best average error, save this as a new best model
            if avg_error < best_error:
                best_model_num = attempt
                best_model = model
                best_error = avg_error
                best_model_errors_list = errors
                best_result = result
            all_errors.append(avg_error)
            attempt += 1

        if self.verbose == 2:
            # Print some stats about the best model
            print("\n\n\n\n\n\nBEST MODEL within the last K folds:\t", best_model_num)
        for error in best_model_errors_list:
            print(error)
        median_error = best_model_errors_list[int(round(len(best_model_errors_list)) / 2)]

        if self.verbose == 2:
            for result in best_result:
                print(result)
        if self.verbose >= 1:
            print("Best median error:", median_error)
            print("Best average error:", best_error)
            print("FINISHED")

        return best_model


if __name__ == "__main__":
    totp = pyotp.TOTP("YHYPKMLAURGON3I2").now()
    rs.login(input("Enter Login Email:\t"), input("Enter Password:\t"), mfa_code=totp)
    ask_model_builder = ModelBuilder(r"data.csv", test_ratio=0.25, verbose=2)
    ask_model = ask_model_builder.make_model(num_folds=7)

    x_test, y_test = ask_model_builder.x_test, ask_model_builder.y_test
    ask_results = ask_model.predict(x_test)
    ask_errors = []
    for i in range(len(y_test)):
        ask_errors.append(1 if evaluate_prediction_quality(ask_results[i], y_test.iloc[i]) else 0)

    avg_ask_error = 1 - sum(ask_errors) / len(ask_errors)
    median_ask_error = ask_errors[int(round(len(ask_errors) / 2))]

    print("Average error:\t", avg_ask_error)
    print("Median error:\t", median_ask_error)

    data_digger = DataDigger(r"temp_data.csv", asset_code='BTC', dT=1, prediction_delay=20)

    log = []
    i = 0
    while True:
        i += 1
        try:
            print(i)
            price_data = data_digger.get_data_row(for_training=False)
            prediction = ask_model.predict(ask_model_builder.reshape_row(price_data))[0]
            log.append((prediction, price_data[-13], price_data[-3]))
            print("Prediction:\t", prediction)
            if len(log) >= 20:
                past_data = log.pop(0)
                print(f"{i-20}'th price and predictions:\t{past_data}")
                past_predictions = past_data[0]
                if past_predictions[0] == past_predictions.max() and past_data[1] < price_data[-13] - 5.0 or \
                    past_predictions[1] == past_predictions.max() and past_data[1] > price_data[-13] + 5.0 or \
                    past_predictions[2] == past_predictions.max() and price_data[-13] - 5.0 < past_data[1] < price_data[-13] + 5.0:
                    print("\u001b[32mGood prediction\u001b[0m\n\n\n")
                else:
                    print("\u001b[31mPoor prediction\u001b[0m\n\n\n")
        except Exception as e:
            print(e)
            print('waited', i, 'times')