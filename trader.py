import turtle, math, pyotp
import robin_stocks.robinhood as rs
from datetime import datetime
from time import sleep
from threading import Thread
from data_handler import ModelBuilder
from data_digger import DataDigger
'''
OLD VERSION
# WHAT IS a Log:
class Log:
    def __init__(self, time, price, crypto_amt):
        # Has time of execution
        self.time = time
        # Has price of order
        self.price = price
        # Has amount of crypto bought/sold in crypto
        self.crypto_total = crypto_amt
        # Has amount of crypto bought/sold in USD
        self.usd_total = crypto_amt*price

class Trader:
    balance = 50
    transactions_log = []
    code = 'BTC'
    price_dT = 5
    self.assets_price = None

    def __init__(self):
        email = 'p.grigorii01@gmail.com'
        password = input("Enter the password")

        totp = pyotp.TOTP("YHYPKMLAURGON3I2").now()
        login = rs.login(email, password, mfa_code=totp)

        self.data_table_update_thread = threading.Thread(target=self.update_data_table,
                                                    args=["coefs.csv", "data.csv", email, password, code])
        self.data_table_update_thread.start()

        self.ask_model_builder = ModelBuilder(file=r"data.csv", dimension='ask', test_ratio=0.3)
        self.ask_model = ask_model_builder.make_model(num_folds=15)

        self.bid_model_builder = ModelBuilder(file=r"data.csv", dimension='bid', test_ratio=0.3)
        self.bid_model = bid_model_builder.make_model(num_folds=15)

        self.model_update_thread = threading.Thread(target=self.update_model)
        self.model_update_thread.start()

        self.balance_offset = float(rs.profiles.load_account_profile()["crypto_buying_power"]) - self.balance
        self.equity_offset = float(self.locate_position_by_code(code)["quantity"])

        self.price_update_thread = threading.Thread(target=self.update_prices)
        self.price_update_thread.start()

        sleep(20)


    def locate_position_by_code(self, pos_code):
        all_equity = rs.crypto.get_crypto_positions()
        position_of_interest = None
        for position in all_equity:
            if position["currency"]["code"] == pos_code:
                position_of_interest = position

        return position_of_interest

    def convert_time_to_seconds(self, time):
        return (int(time[0]) * 60 + int(time[1])) * 60 + int(time[2])

    # HOW TO make an order, KNOWING the ticker, the price and the amount in USD:
    def order_crypto(self, code, price, amount):
        # If the amount is > 0:
        if amount > 0:
            # Send a limit purchase order
            response = rs.order_buy_crypto_limit_by_price (code, amount, price)
        # Else:
        else:
            # Send a limit liquidation order
            response = rs.order_sell_crypto_limit_by_price (code, -amount, price)

        # Return the server response's amount (and -1 if the order is declined)
        print(response)
        try:
            result = float(response["quantity"])
            print(type(result))

            return result
        except Exception:
            print("ORDER DENIED")

            return -1

    # HOW TO reset the avg price:
    def reset_asset_price(self):
        total_USD = 0
        amount = 0
        for record in self.transactions_log:
            if record.crypto_total > 0:
                total_USD += record.crypto_total * record.price
                amount += record.crypto_total
        if amount == 0:
            total_USD = 0
            for i in range (10):
                total_USD += float(rs.get_crypto_quote (self.code, "bid_price"))
                sleep(self.price_dT)
            self.assets_price = total_USD / 10
        else:
            self.assets_price = total_USD / amount
        print("\u001b[36mAvg cost: $" + str(self.assets_price) + "\u001b[0m")


    # Process 1: Update Prices
    def update_prices():
        pass

    # Process 2: Trade
    def trade():
        pass

    # Process 3: Cancel garbage orders
    def cancel_garbage_orders():
        while True:
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
            print("\n")
            sleep(garbage_dT)

    # Process 4: Update the document with coefficients
    def update_data_table(temp_file, destination_file, email, password, code):
        # Forever:
        while True:
            # Run a new DataDigger
            data_digger = DataDigger(temp_file, email, password, code)
            data_digger.fill_data_table(num_trials=1500)
            # Put data from temp_file into destination_file
            destination = open(destination_file, 'w')
            destination.write((line + "\n") for line in data_digger.data_file.readlines())

    # Process 5: Update the estimating model
    def update_model(destination_file):
        global ask_model, ask_model_builder, bid_model, bid_model_builder
        # Forever:
        while True:
            # Make a new ModelBuilder
            ask_model_builder = ModelBuilder(file=destination_file, dimension='ask')
            # Train that ModelBuilder and get the model
            new_model = ask_model_builder.make_model()
            # Assign the new model to the model used elsewhere
            ask_model = new_model

            # Make a new ModelBuilder
            bid_model_builder = ModelBuilder(file=destination_file, dimension='bid')
            # Train that ModelBuilder and get the model
            new_model = bid_model_builder.make_model()
            # Assign the new model to the model used elsewhere
            bid_model = new_model

    # Process 6: Draw the graph of trades

        # Create a turtle for drawing prices

        # Create a turtle for marking executed trades

        # Forever:

            # Position the price-drawing turtle at (current time, current price)

            # If there is a new trade in the log:

                # Position the trade-marking turtle at (order time, order price)

                # Color the turtle to red or green depending on the order type

                # Stamp

                # Move the turtle away
'''

class TradingBot:
    data_digger = None
    def __init__(self, data_file_name, asset_code, refresh_data_table=True, num_trials=1500, num_folds=25, test_ratio=0.15):
        self.data_file_name = data_file_name
        self.temp_file_name = "temp_" + data_file_name

        self.asset_code = asset_code

        if refresh_data_table:
            self.update_table(num_trials=num_trials)

        self.train_models(num_folds=num_folds, test_ratio=test_ratio)

    def update_table (self, num_trials):
        if self.data_digger is not None:
            self.data_digger.data_file.close()

        self.data_digger = DataDigger(self.temp_file_name, input("Enter email: "), input("Enter password: "),
                                      self.asset_code)
        self.data_digger.fill_data_table(num_trials=num_trials)

        temp_data_file = open(self.temp_file_name, 'r')
        data_file = open(self.data_file_name, "w")

        temp_file_contents = temp_data_file.read()
        temp_file_contents = temp_file_contents[:-1]
        data_file.write(temp_file_contents)

        temp_data_file.close()
        data_file.close()

    def train_models (self, num_folds=25, test_ratio=0.15):
        self.buy_model_builder = ModelBuilder(self.data_file_name, "ask",
                                              test_ratio=test_ratio)  # ask === price for which others sell it (program buys it)
        self.sell_model_builder = ModelBuilder(self.data_file_name, "bid",
                                               test_ratio=test_ratio)  # bid === price for which others buy it (program sells it)

        self.buy_model = self.buy_model_builder.make_model(num_folds=num_folds)
        self.sell_model = self.sell_model_builder.make_model(num_folds=num_folds)

def main ():
    bot = TradingBot(r"coefs.csv", 'BTC', num_trials=200)
    # Process 1: analyze data and update the table
    table_update_thread = Thread(target=bot.update_table, kwargs={"num_trials": 700}, daemon=True)
    # Process 2: train the model on the existing data
    model_update_thread = Thread(target=bot.train_models, kwargs={"num_folds": 35, "test_ratio": 0.15})
    # Process 3: update prices, predict and initiate transactions

    while True:
        table_update_thread.start()


if __name__ == "__main__":
    main()