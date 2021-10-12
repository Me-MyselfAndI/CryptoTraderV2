import scipy.misc as spm
import scipy.interpolate as spi
import pyotp, math
from time import sleep
import robin_stocks.robinhood as rs

class DataDigger:
    def __init__(self, file, email, password, asset_code, precision=2, dT=5, coef_depth=10):
        self.code = asset_code
        self.precision = precision
        self.dT = dT
        self.coef_depth=coef_depth

        self.data_file = open(file, 'w')
        self.data_file.write('ask')
        for i in range(coef_depth - 1):
            self.data_file.write('\t' + str(i))

        self.data_file.write('\tbid')
        for i in range(coef_depth - 1):
            self.data_file.write('\t' + str(i))

        self.data_file.write("\task_pade\n")

        totp = pyotp.TOTP("YHYPKMLAURGON3I2").now()
        rs.login(email, password, mfa_code=totp)

    def fill_data_table (self, num_trials):
        factorials = [math.factorial(x) for x in range(self.coef_depth)]
        ask_prices, bid_prices = [], []
        ask_pade_results = [0 for i in range(self.coef_depth * 2)]
        pade_error_log = []

        for trial in range(num_trials):
            trial += 1
            crypto_quote = rs.crypto.get_crypto_quote(self.code)
            ask_prices.append(round(float(crypto_quote["ask_price"]), self.precision))
            bid_prices.append(round(float(crypto_quote["bid_price"]), self.precision))
            print("\nPrices #", trial+1, "\n\tAsk Price: $", ask_prices[-1], "\n\tBid Price: $", bid_prices[-1], sep="")

            sleep(self.dT)

            curr_pade_approximation = 0
            for shift in range(-2, 3):
                curr_pade_approximation += ask_pade_results[self.coef_depth + shift]
            curr_pade_approximation /= 5

            if trial > 4*self.coef_depth + 6:
                pade_error_log.append(curr_pade_approximation-ask_prices[-1])

                total = 0
                for error_amount in pade_error_log:
                    total += abs(error_amount)
                total /= len(pade_error_log)

                print("Average error:", total)

            print("Prediction difference:", curr_pade_approximation-ask_prices[-1])

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

            if len(ask_prices) > 2*self.coef_depth+1:
                ask_taylor_coefs = [spm.derivative(get_ask_price_at, 0, dx=1.0, n=i, order=2*i+1) / factorials[i] for i in range(self.coef_depth)]
                bid_taylor_coefs = [spm.derivative(get_bid_price_at, 0, dx=1.0, n=i, order=2*i+1) / factorials[i] for i in range(self.coef_depth)]
                print(ask_taylor_coefs)
                print(bid_taylor_coefs)

                ask_pade_numerator, ask_pade_denominator = spi.pade(ask_taylor_coefs, math.floor((2 * self.coef_depth - 1) / 4),
                                                                    math.floor((2 * self.coef_depth) / 4))
                bid_pade_numerator, bid_pade_denominator = spi.pade(bid_taylor_coefs, math.floor((2 * self.coef_depth - 1) / 4),
                                                                    math.floor((2 * self.coef_depth) / 4))

                m = 0
                for i in range(len(ask_taylor_coefs)):
                    m += ask_pade_numerator[i] * ((2*self.coef_depth) ** i)

                n = 0
                for i in range(len(ask_taylor_coefs)):
                    n += ask_pade_denominator[i] * ((2*self.coef_depth) ** i)
                ask_pade_results.pop(0)
                ask_pade_results.append(m/n)

                print("Pade approximations: ", curr_pade_approximation)

                temp_counter = 1
                self.data_file.write(str(ask_prices[-1]))
                for i in range(self.coef_depth - 1):
                    temp_counter += 1
                    self.data_file.write('\t' + str(ask_prices[i]))
                self.data_file.write("\t" + str(bid_prices[-1]))
                temp_counter += 1
                for i in range(self.coef_depth - 1):
                    temp_counter += 1
                    self.data_file.write('\t' + str(bid_prices[i]))
                self.data_file.write('\t' + str(ask_pade_results[-1]) + "\n")
                temp_counter += 1
                print(temp_counter)

                ask_prices.pop(0)
                bid_prices.pop(0)

    def close_file(self):
        self.data_file.close()

def main():
    data_digger = DataDigger(r"coefs.csv", input("Enter login email:\t"), input("Enter login password:\t"), 'BTC')
    data_digger.fill_data_table(4000)
    data_digger.close_file()

if __name__ == "__main__":
    main ()