import pandas, warnings
import statsmodels.api as sm

from pmdarima import auto_arima
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

def main ():
    data_frame = pandas.read_csv("coefs.csv", sep='\t', index_col=False).iloc[:-1, :]
    data_frame = pandas.concat([data_frame['ask'],  data_frame["bid"]], axis=1, ignore_index=True)

    for_test = data_frame.iloc[:, 0].dropna()
    print(adfuller(for_test))

    ask_decomposed = sm.tsa.seasonal_decompose(data_frame[0], model='additive', period=100)
    rolling = data_frame.rolling(20)
    plot = ask_decomposed.plot()
    plot.show()

    pyplot.plot(rolling.mean())
    pyplot.plot(rolling.std())
    pyplot.show()


    y_train, y_test = data_frame.iloc[:1200, 0], data_frame.iloc[1200:, 0]
    coefs = auto_arima(y_train, m=15, start_P=0, seasonal=True, d=1, trace=True, suppress_warnings=True, stepwise=True)
    print(coefs)

    model = ARIMA(y_train, order=(1, 0, 15)).fit()
    predictions = model.predict(start=1200, end=1500)
    print(predictions.head())

    y_train.plot(legend=True, label='Training data')
    y_test.plot(legend=True, label='Testing data')
    predictions.plot(legend=True, label='ARIMA')
    pyplot.show()


if __name__ == '__main__':
    main()

