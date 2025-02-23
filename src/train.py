from venv import logger
from src.data_loader import load_complete_data
from src.features import calculate_features
from src.models import HybridModel, StockDataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import pandas as pd



def train_model(symbol, window_size=30, epochs=100):
    # Load and prepare data
    data = load_complete_data(symbol)
    df = calculate_features(symbol)
    
    # Flatten column names consistently
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    
    # Verify Close column exists
    close_col = f'Close_{symbol}'
    if close_col not in df.columns:
        raise KeyError(f"Missing target column {close_col}. Check data loading.")
    
    # Ensure required features exist
    required_tech = ['Returns_', 'Volatility_5D_', 'MA_5_', 'MA_21_', 'Momentum_5D_', 'RSI_', 'MACD_', 'Volume_Change_']
    required_fund = ['pe_ratio_', 'debt_equity_', 'News_Sentiment_', 'Interest_Rate_']
    
    # Handle missing features
    all_required = required_tech + required_fund + [close_col]
    for col in all_required:
        if col not in df.columns:
            df[col] = 0  # Impute missing with zeros
            logger.warning(f"Imputed missing column: {col}")

    # Create features and targets
    tech_data = df[required_tech].values.astype(np.float32)
    fund_data = df[required_fund].values.astype(np.float32)
    targets = df[close_col].values.astype(np.float32)

    # Normalize technical features
    tech_scaler = StandardScaler()
    tech_data = tech_scaler.fit_transform(tech_data)

    # Normalize fundamental features
    fund_scaler = StandardScaler()
    fund_data = fund_scaler.fit_transform(fund_data)

    # Create sequences
    X_tech, X_fund, y = [], [], []
    for i in range(len(tech_data) - window_size):
        X_tech.append(tech_data[i:i+window_size])
        X_fund.append(fund_data[i+window_size-1])
        y.append(targets[i+window_size])

    # Train-validation split (80-20)
    split = int(0.8 * len(X_tech))
    X_tech_train, X_tech_val = X_tech[:split], X_tech[split:]
    X_fund_train, X_fund_val = X_fund[:split], X_fund[split:]
    y_train, y_val = y[:split], y[split:]

    # Create datasets and dataloaders
    train_dataset = StockDataset(X_tech_train, X_fund_train, y_train, window_size)
    val_dataset = StockDataset(X_tech_val, X_fund_val, y_val, window_size)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = HybridModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        train_loss = 0
        for tech, fund, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(tech, fund)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for tech, fund, targets in val_loader:
                outputs = model(tech, fund)
                val_loss += criterion(outputs, targets.view(-1, 1)).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'models/hybrid_model_{symbol}.pth')
            logger.info('Saved best model!')

    logger.info('Training complete!')
    return model