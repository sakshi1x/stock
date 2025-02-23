# from src.data_loader import download_data
from src.predict import predict_stock
import os

from src.train import train_model

def main():
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Download data
    # download_data()
    
    # Generate predictions
    stocks = ["CELH", "CVNA", "UPST", "ALT", "FUBO"]
    # stocks = ["CELH"]
    for symbol in stocks:
        train_model(symbol, epochs=100)
        print(f"Processing {symbol}...")
        predictions = predict_stock(symbol)
        predictions.to_csv(f"data/raw/{symbol}_predictions.csv")
    
    print("All predictions saved in models/ directory")

if __name__ == "__main__":
    main()