import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

SEQ_LEN = 20

class StockDataset(Dataset):
    def __init__(self, df):
        self.data = df.values
        self.X, self.y = [], []
        for i in range(len(self.data) - SEQ_LEN):
            self.X.append(self.data[i:i+SEQ_LEN, :-1])
            self.y.append(self.data[i+SEQ_LEN, 0])  # Predict Close price
        self.X = np.array(self.X)
        self.y = np.array(self.y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

def train_model(train_df, test_df):
    # Extract features and scale target separately
    target_scaler = MinMaxScaler()
    train_df['Target'] = target_scaler.fit_transform(train_df[['Close']])
    test_df['Target'] = target_scaler.transform(test_df[['Close']])

    # Rebuild DataFrame with Target column
    train_df['Target'] = train_df['Target']
    test_df['Target'] = test_df['Target']
    full_train = train_df.drop(columns=['Close']).copy()
    full_train.insert(0, 'Close', train_df['Target'])

    full_test = test_df.drop(columns=['Close']).copy()
    full_test.insert(0, 'Close', test_df['Target'])

    train_dataset = StockDataset(full_train)
    test_dataset = StockDataset(full_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = LSTMModel(input_size=full_train.shape[1]-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(20):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.5f}")

    return model, test_loader, target_scaler
