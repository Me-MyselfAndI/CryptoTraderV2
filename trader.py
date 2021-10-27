import pyotp, robin_stocks.robinhood as rs
from time import sleep
from threading import Thread
from data_handler import ModelBuilder
from data_digger import DataDigger

class TradingBot:
    data_digger = None

    def __init__(self, data_file_name, asset_code, refresh_data_table=True, num_trials=1500, num_folds=10,
                 test_ratio=0.15):
        self.data_file_name = data_file_name
        self.temp_file_name = "temp_" + data_file_name

        self.asset_code = asset_code

        if refresh_data_table:
            self.update_table(num_trials=num_trials)

        self.train_models(num_folds=num_folds, test_ratio=test_ratio)

    def update_table(self, num_trials):
        if self.data_digger is not None:
            self.data_digger.data_file.close()

        self.data_digger = DataDigger(self.temp_file_name, self.asset_code, 20)
        self.data_digger.fill_data_table(num_trials=num_trials)

        temp_data_file = open(self.temp_file_name, 'r')
        data_file = open(self.data_file_name, "w")

        temp_file_contents = temp_data_file.read()
        temp_file_contents = temp_file_contents[:-1]
        data_file.write(temp_file_contents)

        temp_data_file.close()
        data_file.close()

    def train_models(self, num_folds=10, test_ratio=0.15):
        self.buy_model_builder = ModelBuilder(self.data_file_name, "ask",
                                              test_ratio=test_ratio)  # ask === price for which others sell it (program buys it)
        self.sell_model_builder = ModelBuilder(self.data_file_name, "bid",
                                               test_ratio=test_ratio)  # bid === price for which others buy it (program sells it)

        self.buy_model = self.buy_model_builder.make_model(num_folds=num_folds)
        self.sell_model = self.sell_model_builder.make_model(num_folds=num_folds)

    def get_prediction(self):
        curr_data = self.data_digger.get_data_row(for_training=False)

        buy_prediction = self.buy_model.predict(curr_data)
        sell_prediction = self.sell_model.predict(curr_data)

        print(f"\nBuy price prediction: ${buy_prediction}\nSell price prediction: ${sell_prediction}")

        return buy_prediction, sell_prediction

def repeatedly_update_table(bot):
    while True:
        print('\u001b[33m\n\n\nUPDATING TABLE\n\n\n')
        bot.update_table(num_trials=700)
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

    bot = TradingBot(r"coefs.csv", 'BTC', num_trials=100, refresh_data_table=False, num_folds=3)
    # Process 1: analyze data and update the table
    table_update_thread = Thread(target=repeatedly_update_table, kwargs={"bot": bot}, daemon=True)
    table_update_thread.start()
    # Process 2: train the model on the existing data
    model_update_thread = Thread(target=repeatedly_train_models, kwargs={"bot": bot}, daemon=True)
    model_update_thread.start()
    # Process 3: update prices, predict and initiate transactions
    while True:
        input("Predictions are available. Press ENTER to continue")
        print(bot.get_prediction())
        input("Press ENTER to continue")


if __name__ == "__main__":
    main()