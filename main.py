# Old program:
'''
import turtle, math
from datetime import datetime
import robin_stocks.robinhood as rs
import pyotp
from time import sleep
import threading

code = "BSV"
email = 'p.grigorii01@gmail.com'
password = 'Ktyfrhjk#71'

# purchase_curv_threshold    = 0.00004   #
# liquidation_curv_threshold =-0.00004   #

liquidation_price_threshold = -0.1
purchase_price_threshold    = 0.1

sigmoid_smoothness = 0.1
dT = 0.2
averaging_duration    = 20 # make 20
cancel_time_threshold = 80

equity = 0
initial_allowance = 50


class LogRecord:
    # amount is IN DOLLARS. It is positive if bought and negative if sold
    def __init__(self, price, amount, time):
        self.price = price
        self.amount = amount
        self.time = time

    def display(self, in_detail=False):
        if self.amount > 0:
            print("\u001b[32mBought $" + str(self.amount), end="\u001b[0m\t")
        else:
            print("\u001b[31mSold $" + str(-self.amount), end="\u001b[0m\t")
        print("At a price of $" + str(self.price))


def locate_position_by_code(pos_code):
    all_equity = rs.crypto.get_crypto_positions()
    position_of_interest = None
    for position in all_equity:
        if position["currency"]["code"] == pos_code:
            position_of_interest = position

    return position_of_interest


def convert_time_to_seconds(time):
    return (int(time[0]) * 60 + int(time[1])) * 60 + int(time[2])


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
        sleep(10)


def graph():
    global overall_bid_log, overall_ask_log, trans_log
    bid_grapher = turtle.Turtle()
    bid_grapher.speed("fastest")
    bid_grapher.shape("circle")
    bid_grapher.shapesize(.1, .1, .1)
    bid_grapher.pensize(2)
    bid_grapher.penup()
    bid_grapher.setpos(-1000, -1000)

    ask_grapher = turtle.Turtle()
    ask_grapher.speed("fastest")
    ask_grapher.shape("circle")
    ask_grapher.shapesize(.1, .1, .1)
    ask_grapher.pensize(2)
    ask_grapher.penup()
    ask_grapher.setpos(-1000, -1000)

    stamper = turtle.Turtle()
    stamper.speed("fastest")
    stamper.shape("circle")
    stamper.shapesize(.5, .5, .5)
    stamper.pensize(2)
    stamper.penup()
    stamper.setpos(-1000, -1000)

    while not start_drawing.is_set():
        sleep(1)

    time = 0
    start_time = 0
    log_record_num = 0
    start_log_record_num = 0
    while start_drawing.is_set():
        if time < len(overall_bid_log) - 1:
            bid_grapher.color("green")
            bid_grapher.setpos((time - start_time) * 2 - 400, (overall_bid_log[time] - overall_bid_log[start_time]) * 100)
            bid_grapher.pendown()

            ask_grapher.color("red")
            ask_grapher.setpos((time - start_time) * 2 - 400, (overall_ask_log[time] - overall_bid_log[start_time]) * 100) # Says buy, not sell ON PURPOSE! This calibrates the graphs
            ask_grapher.pendown()

            time += 1
        if log_record_num < len(trans_log):
            curr_record = trans_log[log_record_num]
            stamper.color("green" if curr_record.amount > 0 else "red")
            stamper.setpos((curr_record.time - start_time) * 2 - 400,
                           (curr_record.price - overall_bid_log[start_time]) * 100)
            stamper.stamp()
            log_record_num += 1
        if bid_grapher.xcor() > 400:
            bid_grapher.clear()
            ask_grapher.clear()
            start_time = time
            bid_grapher.penup()
            ask_grapher.penup()
            stamper.clearstamps()


totp = pyotp.TOTP("YHYPKMLAURGON3I2").now()
login = rs.login(email, password, mfa_code=totp)

position_of_interest = locate_position_by_code(pos_code=code)

print("Currency:", code, "\tPrice:", rs.crypto.get_crypto_quote(code)["mark_price"])
initial_balance = float(rs.profiles.load_account_profile()["crypto_buying_power"])
initial_equity = float(position_of_interest["quantity"])

if initial_balance < initial_allowance:
    print("Not enough funds allocated in the app. \nNeeded $" + str(initial_allowance),
          "but only $" + str(initial_balance) + " were given")
    quit(0)

balance = initial_allowance
equity = initial_equity

balance_offset = initial_balance - initial_allowance
equity_offset = initial_equity
print("Total Balance: $" + str(initial_balance))
print("Total Equity:", initial_equity)

overall_bid_log = []
overall_ask_log = []
trans_log = []
start_drawing = threading.Event()
start_drawing.clear()
graphing_thread = threading.Thread(target=graph)
graphing_thread.start()

garbage_order_canceller = threading.Thread(target=cancel_garbage_orders)
garbage_order_canceller.start()


assets_price = 0
for i in range(15):
    assets_price += float(rs.crypto.get_crypto_quote(code)["bid_price"])
    sleep(dT)
assets_price /= 15

print(f"Initial price per {code}: $%.4F" % assets_price)

for trial in range(500000):
    # Update stats:
    balance = float(rs.profiles.load_account_profile()["crypto_buying_power"]) - balance_offset
    equity = float(locate_position_by_code(code)["quantity"]) - equity_offset

    held_for_buy = 0
    held_for_sell = 0
    for order in rs.get_all_open_crypto_orders():
        if order["side"] == "sell":
            held_for_sell += float(order["quantity"])
        elif order["side"] == "buy":
            held_for_buy += float(order["quantity"])
        else:
            print("AlErT!")
    start_drawing.set()

    # Get current stock price:
    ask_price = float(rs.crypto.get_crypto_quote(code)["ask_price"])
    bid_price = float(rs.crypto.get_crypto_quote(code)["bid_price"])

    # Adjust buy_coef by sigmoid
    buy_coef = round((balance + equity * bid_price) * (
                1 / (1 + math.exp(-sigmoid_smoothness * (assets_price - bid_price)))), 6)
    sell_coef = round((balance + equity * bid_price) * (
                1 / (1 + math.exp(-sigmoid_smoothness * (assets_price - ask_price)))), 6)

    print("\n\nBuying coefficient: $%.4F" % buy_coef)
    print("Selling coefficient: $%.4F" % sell_coef)
    print("\nBid price: $%.4F" % bid_price)
    print("Ask price: $%.4F" % ask_price)
    print("Equity unit price:\t$%.4F" % assets_price)

    overall_bid_log.append(bid_price)
    overall_ask_log.append(ask_price)



    # If the curvature > than purchase_curv_const and buy_coef > 0, then buy according to the buy_coef
    if buy_coef > 0 and balance > 0:
        trans_log.append(LogRecord(bid_price, min(buy_coef, balance), (trial + 1)))
        balance -= trans_log[-1].amount
        assets_price *= equity
        assets_price += trans_log[-1].amount
        assets_price /= equity + trans_log[-1].amount / trans_log[-1].price
        equity += trans_log[-1].amount / trans_log[-1].price
        response = rs.order_buy_crypto_limit(code, trans_log[-1].amount / trans_log[-1].price, round(trans_log[-1].price, 6))
        try:
            print(round(float(trans_log[-1].amount) / bid_price, 6))
            print(f"\u001b[35mBID PLACED\u001b[0m:" + str(float(response["quantity"]) * float(response["price"])))
        except Exception:
            print(f"\u001b[35mBID ORDER FAILED\u001b[0m $")
            print(response)
            trans_log.pop()


    # Else if the curvature < liquidation_curv_const and buy_coef < 0, then sell according to the buy_coef
    if sell_coef < 0 and equity > 0:
        trans_log.append(LogRecord(ask_price, -min(-sell_coef, equity * ask_price), (trial + 1)))
        balance -= trans_log[-1].amount
        assets_price *= equity
        assets_price -= trans_log[-1].amount
        assets_price /= equity + trans_log[-1].amount / trans_log[-1].price # Plus is on purpose (double negatives)

        equity += trans_log[-1].amount / trans_log[-1].price
        response = rs.order_sell_crypto_limit(code, -trans_log[-1].amount / trans_log[-1].price, round(trans_log[-1].price), 6)
        try:
            print(-round(float(trans_log[-1].amount) / ask_price, 6))
            print(f"\u001b[35mASK PLACED\u001b[0m:" + str(float(response["quantity"]) * float(response["price"])))
        except Exception:
            print(f"\u001b[35mASK ORDER FAILED\u001b[0m $")
            print(response)
            trans_log.pop()
    

    print("\u001b[34m")
    print("Balance: $%.4F" % balance)
    print(f"Equity: {code} %.4F" % equity, "or $: $%.4F" % (equity * bid_price))
    print("Total held on market: $%.4F" % ((held_for_buy + held_for_sell) * bid_price),
          "\tof which for buy: $%.4F" % (held_for_buy * bid_price),
          "and for sell: $%.4F" % (held_for_sell * bid_price))
    print("Total: $%.4F" % ((equity) * bid_price + balance + (held_for_buy + held_for_sell) * bid_price))
    print("\u001b[0m\n")
start_drawing.clear()

print("Total:", equity * bid_price + balance)

'''



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
def order_crypto (code, price, amount):
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
password = 'Ktyfrhjk#71'
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