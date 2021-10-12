import turtle, math, pyotp, threading
import robin_stocks.robinhood as rs
from datetime import datetime
from time import sleep

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


def locate_position_by_code(pos_code):
    all_equity = rs.crypto.get_crypto_positions()
    position_of_interest = None
    for position in all_equity:
        if position["currency"]["code"] == pos_code:
            position_of_interest = position

    return position_of_interest


def convert_time_to_seconds(time):
    return (int(time[0]) * 60 + int(time[1])) * 60 + int(time[2])


# HOW TO make an order, KNOWING the ticker, the price and the amount in USD:
def order_crypto(code, price, amount):
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
def reset_asset_price():
    global assets_price
    total_USD = 0
    amount = 0
    for record in transactions_log:
        if record.crypto_total > 0:
            total_USD += record.crypto_total * record.price
            amount += record.crypto_total
    if amount == 0:
        total_USD = 0
        for i in range (10):
            total_USD += float(rs.get_crypto_quote (code, "bid_price"))
            sleep(price_dT)
        assets_price = total_USD / 10
    else:
        assets_price = total_USD / amount
    print("\u001b[36mAvg cost: $" + str(assets_price) + "\u001b[0m")


# Process 1: Update Prices
def update_prices():
    global buy_price, sell_price, buy_coef, sell_coef, assets_price, balance, equity
    # Forever:
    while True:
        reset_asset_price()
        # Update balance and equity:
        balance = float(rs.profiles.load_account_profile()["crypto_buying_power"]) - balance_offset
        equity = float(locate_position_by_code(code)["quantity"]) - equity_offset
        # Compute the buying and selling prices
        temp_buy_price  = float(rs.get_crypto_quote(code, "bid_price"))
        temp_sell_price = float(rs.get_crypto_quote(code, "ask_price"))
        # Increase the buying price by the maker fee and decrease the selling price by the maker fee
        buy_price  = temp_buy_price  * (1.0000 + maker_fee)
        sell_price = temp_sell_price * (1.0000 - maker_fee)

        # Compute with sigmoid how much should be bought and sold and remember in buying and selling coefficients
        buy_coef = round(balance * (
                1 / (1 + math.exp(-sigmoid_smoothness * (assets_price - buy_price)))), 6)
        sell_coef = round(assets_price * equity * (
                1 / (1 + math.exp(-sigmoid_smoothness * (assets_price - sell_price)))), 6)



        print("\n\n\n\n\n\nassets_price: " + str(assets_price) + "\n\n\n\n\n\n\n")

        # Wait a bit
        sleep(price_dT)

        print("\n\n")
        print (code + " is sold on market at " + str(sell_price))
        print (code + " is bought on market at " + str(buy_price))
        print ("Buying coefficient: " + str(buy_coef))
        print ("Selling coefficient: " + str(sell_coef))


# Process 2: Trade
def trade():
    global buy_price, sell_price, buy_coef, sell_coef, precision, transactions_log, equity, assets_price
    time = 0
    # Forever:
    while True:
        print(buy_price)
        # Remember the current buying and selling prices
        curr_buy_price, curr_sell_price = round(buy_price, precision), round(sell_price, precision)
        curr_buy_coef, curr_sell_coef = buy_coef, sell_coef

        # If the buying coefficient is positive and balance > 0:
        if curr_buy_coef > 0 and balance > 0:
            # Round the buying coefficient according to the allowed precision of the marketplace
            curr_buy_coef = round(curr_buy_coef, precision)
            # Send a limit order to buy an amount according to the minimum(buying coefficient, balance), and by the saved buying price
            response = order_crypto(code, curr_buy_price, min(curr_buy_coef, round(balance, precision)))
            # If the response says the trade was declined:
            if response == -1:
                # print that the transaction was declined
                print("\u001b[34mPurchase declined")
            # Otherwise:
            else:
                # Add the response amount into the log
                transactions_log.append(Log(time, curr_buy_price, response))
                print("\u001b[32mPurchase Order Placed")

        #

        #

        # If the selling coefficient is negative and assets > 0:
        if curr_sell_coef < 0 and equity > 0:
            # Round the selling coefficient according to the allowed precision of the marketplace
            curr_sell_coef = round(curr_sell_coef, precision)
            # Send a limit order to sell an amount according to the maximum(selling coefficient, equity converted into money), and by the saved selling price
            response = order_crypto(code, curr_sell_price, max(curr_sell_coef, -round(assets_price*equity, precision)))
            if response == -1:
                # print that the transaction was declined
                print("\u001b[31mLiquidation declined")
            # Otherwise:
            else:
                # Add the response amount into the log
                transactions_log.append(Log(time, curr_sell_price, -response))
                print("\u001b[33mLiquidation Order Placed")


        held_for_buy = 0
        held_for_sell = 0
        try:
            for order in rs.get_all_open_crypto_orders():
                if order["side"] == "sell":
                    held_for_sell += float(order["quantity"])
                elif order["side"] == "buy":
                    held_for_buy += float(order["quantity"])
                else:
                    print("AlErT!")
        except TypeError:
            pass
        # Print the stats
        print("\u001b[34m")
        print("Balance: $%.4F" % balance)
        print(f"Equity: {code} %.4F" % equity, "or $: $%.4F" % (equity * assets_price))
        print("Total held on market: $%.4F" % ((held_for_buy + held_for_sell) * buy_price),
              "\tof which for buy: $%.4F" % (held_for_buy * buy_price),
              "and for sell: $%.4F" % (held_for_sell * buy_price))
        print("Total: $%.4F" % (equity * assets_price + balance))
        print("\u001b[0m\n")
        # Wait a bit
        sleep(trade_dT)
        #

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

# Process 3: Draw the graph of trades

    # Create a turtle for drawing prices

    # Create a turtle for marking executed trades

    # Forever:

        # Position the price-drawing turtle at (current time, current price)

        # If there is a new trade in the log:

            # Position the trade-marking turtle at (order time, order price)

            # Color the turtle to red or green depending on the order type

            # Stamp

            # Move the turtle away

email = 'p.grigorii01@gmail.com'
password = input("Enter the password")
code = 'BTC'
assets_price = 0
equity = 0
last_prices = []
maker_fee = 0
sigmoid_smoothness = 0.015
price_dT = 0.50
trade_dT = 5
garbage_dT = 1200
precision = 2
transactions_log = []
buy_coef, sell_coef = None, None
buy_price, sell_price = None, None

totp = pyotp.TOTP("YHYPKMLAURGON3I2").now()
login = rs.login(email, password, mfa_code=totp)

balance = 50 #locate_position_by_code(code)
# Update balance and equity:
balance_offset = float(rs.profiles.load_account_profile()["crypto_buying_power"]) - balance
equity_offset = float(locate_position_by_code(code)["quantity"])

price_update_thread = threading.Thread(target=update_prices)
price_update_thread.start()

sleep(20)

trading_thread = threading.Thread(target=trade)
trading_thread.start()

#garbage_collecting_thread = threading.Thread(target=cancel_garbage_orders)
#garbage_collecting_thread.start()
