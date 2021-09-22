import scipy.misc as spm
import scipy.interpolate as spi
import pyotp, math
from time import sleep
import robin_stocks.robinhood as rs

email = 'p.grigorii01@gmail.com'
password = 'Alexpodoksik#66'
code = 'BTC'
precision = 5
dT = 5



totp = pyotp.TOTP("YHYPKMLAURGON3I2").now()
login = rs.login(email, password, mfa_code=totp)


coef_depth = 10
factorials = [math.factorial(x) for x in range(coef_depth)]
ask_prices, bid_prices = [], []
trial = 0
ask_pade_results = [0 for i in range(coef_depth * 2)]
pade_error_log = []

data_file = open("coefs.csv", 'w')
data_file.write('ask')
for i in range (coef_depth - 1):
    data_file.write('\t' + str(i))

data_file.write('\tbid')
for i in range (coef_depth - 1):
    data_file.write('\t' + str(i))

data_file.write("\task_pade")


while True:
    trial += 1
    crypto_quote = rs.crypto.get_crypto_quote(code)
    ask_prices.append(round(float(crypto_quote["ask_price"]), precision))
    bid_prices.append(round(float(crypto_quote["bid_price"]), precision))
    print("\nPrices #", trial+1, "\n\tAsk Price: $", ask_prices[-1], "\n\tBid Price: $", bid_prices[-1], sep="")

    sleep(dT)

    curr_pade_approximation = 0
    for shift in range(-2, 3):
        curr_pade_approximation += ask_pade_results[coef_depth + shift]
    curr_pade_approximation /= 5

    if trial > 4*coef_depth + 6:
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

    if len(ask_prices) > 2*coef_depth+1:
        ask_taylor_coefs = [spm.derivative(get_ask_price_at, 0, dx=1.0, n=i, order=2*i+1) / factorials[i] for i in range(coef_depth)]
        bid_taylor_coefs = [spm.derivative(get_bid_price_at, 0, dx=1.0, n=i, order=2*i+1) / factorials[i] for i in range(coef_depth)]
        print(ask_taylor_coefs)
        print(bid_taylor_coefs)

        ask_pade_numerator, ask_pade_denominator = spi.pade(ask_taylor_coefs, math.floor((2 * coef_depth - 1) / 4),
                                                            math.floor((2 * coef_depth) / 4))
        bid_pade_numerator, bid_pade_denominator = spi.pade(bid_taylor_coefs, math.floor((2 * coef_depth - 1) / 4),
                                                            math.floor((2 * coef_depth) / 4))

        m = 0
        for i in range(len(ask_taylor_coefs)):
            m += ask_pade_numerator[i] * ((2*coef_depth) ** i)

        n = 0
        for i in range(len(ask_taylor_coefs)):
            n += ask_pade_denominator[i] * ((2*coef_depth) ** i)
        ask_pade_results.pop(0)
        ask_pade_results.append(m/n)

        print("Pade approximations: ", curr_pade_approximation)

        temp_counter = 1
        data_file.write(str(ask_prices[-1]))
        for i in range(coef_depth - 1):
            temp_counter += 1
            data_file.write('\t' + str(ask_prices[i]))
        data_file.write("\t" + str(bid_prices[-1]))
        temp_counter += 1
        for i in range(coef_depth - 1):
            temp_counter += 1
            data_file.write('\t' + str(bid_prices[i]))
        data_file.write('\t' + str(ask_pade_results[-1]))
        temp_counter += 1
        print(temp_counter)

        ask_prices.pop(0)
        bid_prices.pop(0)
