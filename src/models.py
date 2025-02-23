from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np

class ProphetModel:
    def __init__(self):
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        
    def train(self, df):
        train_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        self.model.fit(train_df)
        
    def predict(self, days=30):
        future = self.model.make_future_dataframe(periods=days)
        return self.model.predict(future)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x

class EnsembleModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf = RandomForestRegressor(n_estimators=100)
        
    def prepare_features(self, df):
        features = df[['Volatility_5D', 'MA_5', 'MA_21', 'Momentum_5D']]
        return self.scaler.fit_transform(features)
    
    def train(self, X, y):
        self.rf.fit(X, y)
        
    def predict(self, X):
        return self.rf.predict(X)
   
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, tech_features, fund_features, targets, window_size=30):
        self.window_size = window_size
        self.tech_features = tech_features  # Should be 3D: (num_samples, window_size, num_technical)
        self.fund_features = fund_features  # Should be 2D: (num_samples, num_fundamental)
        self.targets = targets              # Should be 1D: (num_samples,)

    def __len__(self):
        return len(self.tech_features)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.tech_features[idx]),  # (window_size, num_technical)
            torch.FloatTensor(self.fund_features[idx]),  # (num_fundamental,)
            torch.FloatTensor([self.targets[idx]])       # (1,)
        )

    
class HybridModel(nn.Module):
    def __init__(self, num_technical=8, num_fundamental=4):
        super().__init__()
        
        # Technical analysis branch
        self.tech_lstm = nn.LSTM(
            input_size=num_technical,
            hidden_size=32,
            batch_first=True  # Expects (batch, seq_len, features)
        )
        
        # Fundamental branch (4 features without sentiment)
        self.fund_dense = nn.Sequential(
            nn.Linear(num_fundamental, 16),
            nn.ReLU()
        )
        
        # Combined prediction
        self.combined = nn.Linear(32 + 16, 1)

    def forward(self, tech_features, fund_features):
        # Fix dimensions if needed
        if tech_features.dim() == 4:
            tech_features = tech_features.squeeze(1)
            
        # Technical analysis
        tech_out, _ = self.tech_lstm(tech_features)  # (batch, seq_len, 32)
        tech_out = tech_out[:, -1, :]                # (batch, 32)
        
        # Fundamental analysis
        fund_out = self.fund_dense(fund_features)    # (batch, 16)
        
        # Combine features
        combined = torch.cat([tech_out, fund_out], dim=1)  # (batch, 48)
        return self.combined(combined)                      # (batch, 1)