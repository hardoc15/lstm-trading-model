import numpy as np
import matplotlib.pyplot as plt

def simulate_trading(preds, actuals, initial_cash=10000):
    cash = initial_cash
    position = 0
    portfolio = []

    for i in range(len(preds) - 1):
        pred_today = preds[i]
        actual_today = actuals[i]
        actual_next = actuals[i + 1]

        if pred_today > actual_today:
            if position == 0:
                position = cash // actual_today
                cash -= position * actual_today
        else:
            cash += position * actual_today
            position = 0

        total_value = cash + position * actual_today
        portfolio.append(total_value)

    return portfolio

def plot_backtest(portfolio_values):
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_values, label='LSTM Strategy')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Backtest Results')
    plt.legend()
    plt.grid(True)
    plt.show()
