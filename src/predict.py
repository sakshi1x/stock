from datetime import datetime

import warnings
import pandas as pd

from src.data_loader import load_complete_data
from src.models import HybridModel, ProphetModel, LSTMModel, EnsembleModel
from .features import calculate_features
import torch

from dotenv import load_dotenv
import os

load_dotenv()
warnings.filterwarnings('ignore')

PREDICTION_DAYS = 30
START_DATE = os.getenv('START_DATE')
END_DATE = os.getenv('END_DATE')


def predict_stock(symbol):
    # Load all data types
    data = load_complete_data(symbol)
    
    
    # Prepare features
    df = calculate_features(symbol)

    # Flatten column names if MultiIndex
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    # print("Available columns:", df.columns)  # Debugging

    # Ensure required columns exist before selection
    required_fund_features = ['pe_ratio_', 'debt_equity_', 'News_Sentiment_', 'Interest_Rate_']
    missing_fund_features = [col for col in required_fund_features if col not in df.columns]

    if missing_fund_features:
        print(f"⚠️ Missing fundamental features: {missing_fund_features}")
        fund_features = None  # Handle missing features gracefully
    else:
        fund_features = df[required_fund_features].values

    # Create technical features
    tech_features = df[[
        'Returns_', 'Volatility_5D_', 'MA_5_', 'MA_21_', 
        'Momentum_5D_', 'RSI_', 'MACD_', 'Volume_Change_'
    ]].values

    # Prepare PyTorch tensors
    tech_tensor = torch.FloatTensor(tech_features).unsqueeze(0)

    # Handle empty fundamental features safely
    if fund_features is None or len(fund_features) == 0:
        print("⚠️ No fundamental features available, using zeros.")
        fund_tensor = torch.zeros((1, len(required_fund_features)))  # Use zero tensor
    else:
        fund_tensor = torch.FloatTensor(fund_features[-1]).unsqueeze(0)
    model_path = f"models/hybrid_model_{symbol}.pth"
    
    try:
        model = HybridModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except FileNotFoundError:
        raise Exception(f"Model not found at {model_path}. Train first!")
    volatility = df['Volatility_30D'].iloc[-1] if 'Volatility_30D' in df.columns else 0

    # Make prediction
    with torch.no_grad():
        prediction = model(tech_tensor, fund_tensor).item()

        result = pd.DataFrame({
        'symbol': [symbol],
        'date': [START_DATE],
        'end date':[END_DATE],
        'prediction': [prediction],
        'confidence': [abs(prediction)] , # Example metric


    })
    
    return result



def save_predictions(predictions, symbol):
    predictions.to_csv(f"models/{symbol}_predictions.csv")