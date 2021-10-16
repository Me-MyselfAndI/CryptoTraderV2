import scipy.misc as spm
import scipy.interpolate as spi
import pyotp, math
from time import sleep
import robin_stocks.robinhood as rs

class DataDigger:
    def __init__(self, file_name, email, password, asset_code, precision=2, dT=5, coef_depth=10):
        self.code = asset_code
        self.precision = precision
        self.dT = dT
        self.coef_depth=coef_depth
        self.file_name = file_name

        totp = pyotp.TOTP("YHYPKMLAURGON3I2").now()
        rs.login(email, password, mfa_code=totp)

    def fill_data_table (self, num_trials, prediction_delay=20):
        self.data_file = open(self.file_name, 'w')
        self.data_file.write('ask')
        for i in range(self.coef_depth - 1):
            self.data_file.write('\t' + str(i))

        self.data_file.write('\tbid')
        for i in range(self.coef_depth - 1):
            self.data_file.write('\t' + str(i))

        self.data_file.write("\task_pade\tbid_pade\n")

        # Needed to avoid recalculating factorials in computations of Taylor polynomial coefficients
        factorials = [math.factorial(x) for x in range(self.coef_depth)]

        # most recent ask_prices and bid_prices
        ask_prices, bid_prices = [], []

        # Predictions by Pade approximation
        ask_pade_predictions = [0 for i in range(self.coef_depth * 2 + 20)]
        bid_pade_predictions = [0 for i in range(self.coef_depth * 2 + 20)]

        # Initial number of predictions are very wrong, since they
        # depend on having previously collected all the data
        threshold_to_start_filling_table = 2 * self.coef_depth + 1

        # Start predictions
        for trial in range(1, num_trials + threshold_to_start_filling_table + 1):
            # Get the prices
            crypto_quote = rs.crypto.get_crypto_quote(self.code)
            ask_prices.append(round(float(crypto_quote["ask_price"]), self.precision))
            bid_prices.append(round(float(crypto_quote["bid_price"]), self.precision))
            print("\nPrices #", trial+1, "\n\tAsk Price: $", ask_prices[-1], "\n\tBid Price: $", bid_prices[-1], sep="")

            # Find the Pade Approximation for the current price
            curr_pade_approximation = {'ask': 0, 'bid': 0}
            for shift in range(-2, 3):
                curr_pade_approximation['ask'] += ask_pade_predictions[self.coef_depth + shift]
                curr_pade_approximation['bid'] += bid_pade_predictions[self.coef_depth + shift]
            curr_pade_approximation['ask'] /= 5
            curr_pade_approximation['bid'] /= 5

            print(f"Prediction difference:\n\tAsk: ${curr_pade_approximation['ask']-ask_prices[-1]}\n\tBid: ${curr_pade_approximation['bid']-bid_prices[-1]}")

            # Functions that get ask and bid prices
            # Needed since Pade requires a continious function, and the price cannot
            # be obtained at non-integer indices of the list
            def get_ask_price_at(x):
                below, above = ask_prices[math.floor(x)], ask_prices[math.ceil(x)]
                if math.ceil(x) == x:
                    return above
                return (above - below) / (math.ceil(x) - math.floor(x)) * (x - math.floor(x)) + below

            def get_bid_price_at(x):
                below, above = bid_prices[math.floor(x)], bid_prices[math.ceil(x)]
                if math.ceil(x) == x:
                    return above
                return (above - below) / (math.ceil(x) - math.floor(x)) * (x - math.floor(x)) + below

            #
            if len(ask_prices) > threshold_to_start_filling_table + prediction_delay:
                ask_taylor_coefs = [spm.derivative(get_ask_price_at, 0, dx=1.0, n=i, order=2*i+1) / factorials[i] for i in range(self.coef_depth)]
                bid_taylor_coefs = [spm.derivative(get_bid_price_at, 0, dx=1.0, n=i, order=2*i+1) / factorials[i] for i in range(self.coef_depth)]
                print(ask_taylor_coefs)
                print(bid_taylor_coefs)

                # Pade approximation is a fraction with polynomials in both enumerator and denominator
                # Creating those polinomials for ask and bid prices
                ask_pade_numerator, ask_pade_denominator = spi.pade(ask_taylor_coefs, math.floor((2 * self.coef_depth - 1) / 4),
                                                                    math.floor((2 * self.coef_depth) / 4))
                bid_pade_numerator, bid_pade_denominator = spi.pade(bid_taylor_coefs, math.floor((2 * self.coef_depth - 1) / 4),
                                                                    math.floor((2 * self.coef_depth) / 4))


                # Converting numerator and denominator of ask Pade approx. from polinomials to values at the required point
                numerator_value = 0
                for i in range(len(ask_taylor_coefs)):
                    numerator_value += ask_pade_numerator[i] * ((2 * self.coef_depth) ** i)

                denominator_value = 0
                for i in range(len(ask_taylor_coefs)):
                    denominator_value += ask_pade_denominator[i] * ((2 * self.coef_depth) ** i)
                ask_pade_predictions.pop(0)
                ask_pade_predictions.append(numerator_value / denominator_value)

                # Converting numerator and denominator of bid Pade approx. from polinomials to values at the required point
                numerator_value = 0
                for i in range(len(bid_taylor_coefs)):
                    numerator_value += bid_pade_numerator[i] * ((2 * self.coef_depth) ** i)

                denominator_value = 0
                for i in range(len(bid_taylor_coefs)):
                    denominator_value += bid_pade_denominator[i] * ((2 * self.coef_depth) ** i)
                bid_pade_predictions.pop(0)
                bid_pade_predictions.append(numerator_value / denominator_value)

                print(f"Pade approximations:\n\tAsk: ${ask_pade_predictions[-1]}\n\tBid: ${bid_pade_predictions[-1]}")


                # Write everything into the data table
                temp_counter = 1
                self.data_file.write(str(ask_prices[-prediction_delay - 1]))
                for i in range(self.coef_depth - 1):
                    temp_counter += 1
                    self.data_file.write('\t' + str(ask_prices[i]))
                self.data_file.write("\t" + str(bid_prices[-prediction_delay - 1]))
                temp_counter += 1
                for i in range(self.coef_depth - 1):
                    temp_counter += 1
                    self.data_file.write('\t' + str(bid_prices[i]))
                self.data_file.write('\t' + str(ask_pade_predictions[-prediction_delay]) + '\t' + str(bid_pade_predictions[-prediction_delay]) + "\n")
                temp_counter += 2

                ask_prices.pop(0)
                bid_prices.pop(0)

            sleep(self.dT)


        self.data_file.close()

def main():
    data_digger = DataDigger(r"temp_coefs.csv", input("Enter login email:\t"), input("Enter login password:\t"), 'BTC')
    data_digger.fill_data_table(1000)

if __name__ == "__main__":
    main ()