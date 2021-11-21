import scipy.misc as spm
import scipy.interpolate as spi
import pyotp, math
from time import sleep
import robin_stocks.robinhood as rs



class DataDigger:
    def __init__(self, file_name, asset_code, prediction_delay, precision=5, dT=5, coef_depth=10):
        self.code = asset_code
        self.precision = precision
        self.dT = dT
        self.coef_depth=coef_depth
        self.file_name = file_name
        self.prediction_delay = prediction_delay

        # Needed to avoid recalculating factorials in computations of Taylor polynomial coefficients
        self._factorials = [math.factorial(x) for x in range(self.coef_depth)]


        # most recent self.ask_prices and self.bid_prices
        self.ask_prices, self.bid_prices = [], []

        # Predictions by Pade approximation
        self.ask_pade_predictions = [0 for i in range(self.coef_depth * 2 + self.prediction_delay + 1)]
        self.bid_pade_predictions = [0 for i in range(self.coef_depth * 2 + self.prediction_delay + 1)]


        # Initial number of predictions are very wrong, since they
        # depend on having previously collected all the data
        self._threshold_to_start_filling_table = 2 * self.coef_depth + 1

        self.fill_data_table(self._threshold_to_start_filling_table)


    def get_data_row(self, for_training=True):
        # Get the prices
        crypto_quote = rs.crypto.get_crypto_quote(self.code)
        self.ask_prices.append(round(float(crypto_quote["ask_price"]), self.precision))
        self.bid_prices.append(round(float(crypto_quote["bid_price"]), self.precision))
        print(f"\tAsk Price: ${round(float(crypto_quote['ask_price']), self.precision)}\n"
              f"\tBid Price: ${round(float(crypto_quote['bid_price']), self.precision)}")

        # Functions that get ask and bid prices
        # Needed since Pade requires a continious function, and the price cannot
        # be obtained at non-integer indices of the list
        def get_ask_price_change_at(x):
            below, above = self.ask_prices[math.floor(x+1)] - self.ask_prices[math.floor(x)], \
                           self.ask_prices[math.ceil(x+1)] - self.ask_prices[math.ceil(x)]
            if math.ceil(x) == x:
                return above
            return (above - below) / (math.ceil(x) - math.floor(x)) * (x - math.floor(x)) + below

        def get_bid_price_change_at(x):
            below, above = self.bid_prices[math.floor(x+1)] - self.bid_prices[math.floor(x)], \
                           self.bid_prices[math.ceil(x+1)] - self.bid_prices[math.ceil(x)]
            if math.ceil(x) == x:
                return above
            return (above - below) / (math.ceil(x) - math.floor(x)) * (x - math.floor(x)) + below

        sleep(self.dT)

        if len(self.ask_prices) > self._threshold_to_start_filling_table + self.prediction_delay:
            ask_taylor_coefs = [spm.derivative(get_ask_price_change_at, 0, dx=1.0, n=i, order=2 * i + 1) / self._factorials[i]
                                for i in range(self.coef_depth)]
            bid_taylor_coefs = [spm.derivative(get_bid_price_change_at, 0, dx=1.0, n=i, order=2 * i + 1) / self._factorials[i]
                                for i in range(self.coef_depth)]
            print(ask_taylor_coefs)
            print(bid_taylor_coefs)

            # Pade approximation is a fraction with polynomials in both enumerator and denominator
            # Creating those polinomials for ask and bid prices
            ask_pade_numerator, ask_pade_denominator = spi.pade(ask_taylor_coefs,
                                                                math.floor((2 * self.coef_depth - 1) / 4),
                                                                math.floor((2 * self.coef_depth) / 4))
            bid_pade_numerator, bid_pade_denominator = spi.pade(bid_taylor_coefs,
                                                                math.floor((2 * self.coef_depth - 1) / 4),
                                                                math.floor((2 * self.coef_depth) / 4))

            # Converting numerator and denominator of ask Pade approx. from polinomials to values at the required point
            numerator_value = 0
            for i in range(len(ask_taylor_coefs)):
                numerator_value += ask_pade_numerator[i] * ((2 * self.coef_depth) ** i)

            denominator_value = 0
            for i in range(len(ask_taylor_coefs)):
                denominator_value += ask_pade_denominator[i] * ((2 * self.coef_depth) ** i)
            curr_ask_pade_prediction = numerator_value / denominator_value
            self.ask_pade_predictions.pop(0)
            self.ask_pade_predictions.append(curr_ask_pade_prediction)

            # Converting numerator and denominator of bid Pade approx. from polinomials to values at the required point
            numerator_value = 0
            for i in range(len(bid_taylor_coefs)):
                numerator_value += bid_pade_numerator[i] * ((2 * self.coef_depth) ** i)

            denominator_value = 0
            for i in range(len(bid_taylor_coefs)):
                denominator_value += bid_pade_denominator[i] * ((2 * self.coef_depth) ** i)
            curr_bid_pade_prediction = numerator_value / denominator_value
            self.bid_pade_predictions.pop(0)
            self.bid_pade_predictions.append(curr_bid_pade_prediction)

            self.ask_prices.pop(0)
            self.bid_prices.pop(0)

            if for_training:
                ask_return_data = [self.ask_prices[-self.prediction_delay - 1]]
                ask_return_data.extend(self.ask_prices[:self.coef_depth-1])
                bid_return_data = [self.bid_prices[-self.prediction_delay - 1]]
                bid_return_data.extend(self.bid_prices[:self.coef_depth-1])
                pade_predictions_return = [self.ask_pade_predictions[-self.prediction_delay], self.bid_pade_predictions[-self.prediction_delay]]
            else:
                ask_return_data = self.ask_prices[-self.coef_depth:-1]
                bid_return_data = self.bid_prices[-self.coef_depth:-1]
                pade_predictions_return = [self.ask_pade_predictions[-self.prediction_delay], self.bid_pade_predictions[-self.prediction_delay]]

            result = ask_return_data + bid_return_data + pade_predictions_return
            return result

        print(f"\u001b[31mNOT ENOUGH PAST DATA TO GENERATE PADE APPROXIMATIONS. Run "
              f"get_one_row() {self._threshold_to_start_filling_table + self.prediction_delay - len(self.ask_prices) + 1}"
              f" more times to start getting predictions\u001b[0m")
        return False

    def fill_data_table (self, num_trials):
        self.data_file = open(self.file_name, 'w')
        self.data_file.write('ask')
        for i in range(self.coef_depth - 1):
            self.data_file.write('\ta' + str(i))

        self.data_file.write('\tbid')
        for i in range(self.coef_depth - 1):
            self.data_file.write('\tb' + str(i))

        self.data_file.write("\task_pade\tbid_pade\n")

        # Start predictions
        for trial in range(1, num_trials + self._threshold_to_start_filling_table + 1):
            print (f"\nPrices #{trial + 1}")
            last_row_data = self.get_data_row()
            if last_row_data and trial > self.coef_depth:
                # Write everything into the data table
                temp_counter = 1
                self.data_file.write(str(self.ask_prices[-self.prediction_delay - 1]))
                for i in range(self.coef_depth - 1):
                    temp_counter += 1
                    self.data_file.write('\t' + str(self.ask_prices[i]))
                self.data_file.write("\t" + str(self.bid_prices[-self.prediction_delay - 1]))
                temp_counter += 1
                for i in range(self.coef_depth - 1):

                    temp_counter += 1
                    self.data_file.write('\t' + str(self.bid_prices[i]))
                self.data_file.write('\t' + str(self.ask_pade_predictions[-self.prediction_delay]) + '\t' +
                                     str(self.bid_pade_predictions[-self.prediction_delay]) + "\n")
                temp_counter += 2
                print(last_row_data)

        self.data_file.close()


def main():
    email = input("Enter login email:\t")
    password = input("Enter login password:\t")

    totp = pyotp.TOTP("YHYPKMLAURGON3I2").now()
    rs.login(email, password, mfa_code=totp)

    data_digger = DataDigger(r"coefs.csv", asset_code='BTC', prediction_delay=100)
    data_digger.fill_data_table(500)

if __name__ == "__main__":
    main ()