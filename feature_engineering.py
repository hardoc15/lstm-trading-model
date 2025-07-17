import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def feature_engineering(df):
    df['Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['Volume_Change'] = df['Volume'].pct_change()

    df = df.dropna()

    features = ['Close', 'Return', 'MA_5', 'MA_10', 'Volatility', 'Volume_Change']
    df = df[features]

    # Normalize
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df
