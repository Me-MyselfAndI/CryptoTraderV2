import numpy
import pyotp, robin_stocks.robinhood as rs
from time import sleep
from threading import Thread
from _datetime import datetime

from robin_stocks.orders import order_crypto

from data_handler import ModelBuilder
from data_digger import DataDigger

def locate_position_by_code(pos_code):
    all_equity = rs.crypto.get_crypto_positions()
    position_of_interest = None
    for position in all_equity:
        if position["currency"]["code"] == pos_code:
            position_of_interest = position

    return position_of_interest


def convert_time_to_seconds(time):
    return (int(time[0]) * 60 + int(time[1])) * 60 + int(time[2])

def cancel_garbage_orders(cancel_time_threshold, garbage_dT):
    while True:
        try:
            all_orders = rs.get_all_open_crypto_orders()
            print("\t")
            if all_orders:
                for i in range(len(all_orders)):
                    order = all_orders[i]
                    order_time = order["created_at"].replace('T', '.')
                    order_time = order_time.split(sep='.')[1].split(':')

                    real_time = datetime.now().strftime('%H:%M:%S').split(sep=":")

                    if convert_time_to_seconds(real_time) - convert_time_to_seconds(order_time) > cancel_time_threshold:
                        print("\u001b[33mCancelling", i + 1, "-", order_time, "\u001b[0m")
                        rs.cancel_crypto_order(order["id"])
        except Exception as e:
            print(f"Problems cancelling garbage orders: {e}")
        print("\n")
        sleep(garbage_dT)
class TradingBot:
    data_digger = None

    def __init__(self, data_file_name, asset_code, num_trials=150, num_folds=10, test_ratio=0.15):
        self.data_file_name = data_file_name
        self.temp_file_name = "temp_" + data_file_name

        self.asset_code = asset_code

        self.update_table(num_trials=num_trials)

        self.train_models(num_folds=num_folds, test_ratio=test_ratio)

    def update_table(self, num_trials):
        if self.data_digger is not None:
            self.data_digger.data_file.close()

        new_data_digger = DataDigger(self.temp_file_name, self.asset_code, dT=1, prediction_delay=20)
        new_data_digger.fill_data_table(num_trials=num_trials)

        temp_data_file = open(self.temp_file_name, 'r')
        data_file = open(self.data_file_name, "w")

        temp_file_contents = temp_data_file.read()
        temp_file_contents = temp_file_contents[:-1]
        data_file.write(temp_file_contents)

        temp_data_file.close()
        data_file.close()

        self.data_digger = new_data_digger

    def train_models(self, num_folds=10, test_ratio=0.15):
        self.model_builder = ModelBuilder(self.data_file_name, test_ratio=test_ratio)
        self.model = self.model_builder.make_model(num_folds=num_folds)

    def get_prediction(self):
        curr_data = self.data_digger.get_data_row(for_training=False)
        curr_data = self.model_builder.reshape_row(curr_data)
        try:
            buy_prediction = self.model.predict(curr_data)
        except Exception as exception:
            print(exception)
            return False
        print(f"\nBuy price prediction:\t${buy_prediction[0][0]}")

        return buy_prediction

def repeatedly_update_table(bot):
    while True:
        print('\u001b[33m\n\n\nUPDATING TABLE\n\n\n')
        bot.update_table(num_trials=500)
        print("\n\n\nFINISHED UPDATING TABLE\u001b[0m\n\n\n")


def repeatedly_train_models(bot):
    while True:
        print('\u001b[32m\n\n\nRETRAINING MODEL\n\n\n')
        bot.train_models(num_folds=15, test_ratio=0.15)
        print("\n\n\n\FINISHED RETRAINING MODEL\u001b[0m\n\n\n")


def main():
    email = input("Enter login email:\t")
    password = input("Enter login password:\t")

    totp = pyotp.TOTP("YHYPKMLAURGON3I2").now()
    rs.login(email, password, mfa_code=totp)

    prices_log = []

    asset_code = 'BTC'
    minimal_sell_balance = 10
    initial_available_balance = 50
    balance_offset = float(rs.profiles.load_account_profile()["crypto_buying_power"]) - initial_available_balance
    initial_available_equity = 0
    equity_offset = float(locate_position_by_code(asset_code)["quantity"]) - initial_available_equity

    garbage_dT = 2
    cancel_time_threshold = 150

    # Process 0: cancel expired orders
    cancel_garbage_orders_thread = Thread(target=cancel_garbage_orders, daemon=True,
                                          kwargs={'cancel_time_threshold': cancel_time_threshold, 'garbage_dT': garbage_dT})
    cancel_garbage_orders_thread.start()

    bot = TradingBot(r"coefs.csv", asset_code, num_trials=1000, num_folds=20)
    # Process 1: analyze data and update the table
    table_update_thread = Thread(target=repeatedly_update_table, kwargs={"bot": bot}, daemon=True)
    table_update_thread.start()

    # Process 2: train the model on the existing data
    model_update_thread = Thread(target=repeatedly_train_models, kwargs={"bot": bot}, daemon=True)
    model_update_thread.start()

    # Process 3: update prices, predict and initiate transactions
    def make_trade(balances):
        verdict_buy, verdict_sell = 'no', 'no'
        max_prediction = max(predictions['buy'], predictions['sell'], predictions['hold'])
        avg_prediction = (predictions['buy'] + predictions['sell'] + predictions['hold'])/3
        if predictions['buy'] == max_prediction and predictions['buy'] > avg_prediction * 1.25 and balances['USD'] > minimal_sell_balance:
            # buy
            response = rs.order_buy_crypto_by_price(symbol=asset_code, amountInDollars=round(balances['USD'] / 3, 2))
            verdict_buy = 'yes'
        if predictions['sell'] == max_prediction and predictions['sell'] > avg_prediction * 1.25 and balances[asset_code] > 0:
            # sell
            response = rs.order_sell_crypto_by_quantity(symbol=asset_code, quantity=balances[asset_code],)
            verdict_sell = 'yes'

        #old_price = curr_price['ask'], curr_price['bid']
        price_prediction = predictions
        sleep(bot.data_digger.dT)
        #ask_price_new, bid_price_new = curr_price['ask'], curr_price['bid']
        prices_log.append({
            #'old_price' : old_price,
            'prediction': price_prediction,
            'bought': verdict_buy, 'sold': verdict_sell
        })

    while True:
        prediction_response = bot.get_prediction()[0]
        #curr_price = {'ask': prediction_response[0], 'bid': prediction_response[1]}
        predictions = {'buy': prediction_response[0], 'sell': prediction_response[1], 'hold': prediction_response[2]}
        print(f"\u001b[34mPredictions are available:\n{bot.get_prediction()}\u001b[0m")

        balances = {'USD': float(rs.profiles.load_account_profile()["crypto_buying_power"]) - balance_offset, asset_code: float(locate_position_by_code(asset_code)["quantity"]) - equity_offset}
        print(f"Balance: ${balances['USD']}, BTC {balances[asset_code]}")
        trade_thread = Thread(target=make_trade, args=[balances])
        trade_thread.start()
        sleep(bot.data_digger.dT)
        for log_item in prices_log:
            print(log_item)

if __name__ == "__main__":
    main()