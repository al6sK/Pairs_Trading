import yfinance as yf

startDateStr = '2007-12-01'
endDateStr = '2017-12-01'

instrumentIds = ['SPY','AAPL','ADBE','EBAY','MSFT','QCOM','HPQ','AMD','IBM']


for id in instrumentIds:
    data=yf.Ticker(id)
    df = data.history(start=startDateStr, end=endDateStr , interval="1d", actions=True)

    df['SplitFactor'] = df['Stock Splits'].replace(0, 1).cumprod()
    df['Adj Close'] = df['Close'] / df['SplitFactor']
    df['Adj Close'] = df['Adj Close'] - df['Dividends'].cumsum()
    df['Adj Close'].to_csv(f"DATA/{id}.csv")
