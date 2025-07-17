import pandas as pd
from data import load_stock_data
from feature_engineering import feature_engineering
from train import train_model
from evaluate import evaluate_model

# Step 1: Load data
df = load_stock_data('AAPL', start='2015-01-01', end='2023-12-31')

# Step 2: Feature engineering
df = feature_engineering(df)

# Step 3: Split data
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# Step 4: Train model
model, test_loader, target_scaler = train_model(train_data, test_data)

# Step 5: Evaluate model
preds, actuals = evaluate_model(model, test_loader, target_scaler)

# Step 6: Visualize results
import matplotlib.pyplot as plt
plt.plot(actuals, label='Actual')
plt.plot(preds, label='Predicted')
plt.legend()
plt.title('Model Predictions vs Actual')
plt.show()
