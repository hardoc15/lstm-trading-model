import yfinance as yf
import pandas as pd

def load_stock_data(ticker='AAPL', start='2018-01-01', end='2023-12-31'):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df
