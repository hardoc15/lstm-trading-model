import torch
import numpy as np

def evaluate_model(model, test_loader, target_scaler):
    model.eval()
    all_preds, all_actuals = [], []

    with torch.no_grad():
        for X, y in test_loader:
            preds = model(X).numpy()
            all_preds.extend(preds)
            all_actuals.extend(y.numpy())

    all_preds = np.array(all_preds).reshape(-1, 1)
    all_actuals = np.array(all_actuals).reshape(-1, 1)

    # Inverse transform to original scale
    preds = target_scaler.inverse_transform(all_preds).flatten()
    actuals = target_scaler.inverse_transform(all_actuals).flatten()
    return preds, actuals
