from venv import logger
from src.data_loader import get_news_with_dates, load_complete_data
from textblob import TextBlob
import yfinance as yf
import requests
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from fredapi import Fred 
from datetime import datetime, timedelta
import pandas as pd
import nltk
from dotenv import load_dotenv
import os

load_dotenv()
nltk.download('vader_lexicon')


FRED_API_KEY=os.getenv('FRED_API_KEY')
NEWS_API_KEY= os.getenv('NEWS_API_KEY')


PREDICTION_DAYS = 30
WINDOW_SIZE = 30
START_DATE = os.getenv('START_DATE')
END_DATE = os.getenv('END_DATE')

NEWS_API_LOOKBACK_DAYS = 29 


def add_sentiment_features(df, news_data):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    
    if isinstance(news_data, pd.DataFrame) and not news_data.empty:
        for _, row in news_data.iterrows():
            article = row.get('news', {})
            title = article.get('title', '')
            summary = article.get('summary', '')
            
            if title or summary:
                text = f"{title} {summary}"
                blob = TextBlob(text)
                vader_score = sia.polarity_scores(text)['compound']
                combined = (blob.sentiment.polarity + vader_score) / 2
                sentiment_scores.append(combined)
    
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    df['News_Sentiment'] = avg_sentiment
    df['Social_Sentiment'] = np.random.uniform(-1, 1, len(df))  # Placeholder
    
    print(f"Processed {len(sentiment_scores)} articles. Avg Sentiment: {avg_sentiment}")
    return df

def add_macro_features(df):
    # Get real economic data (example using FRED)
    fred = Fred(api_key=FRED_API_KEY)
    
    df['Interest_Rate'] = fred.get_series('DFF').iloc[-1]
    df['Inflation'] = fred.get_series('CPIAUCSL').pct_change(12).iloc[-1]
    df['VIX'] = yf.download('^VIX', period='1d')['Close'].iloc[-1]
    
    return df








def calculate_technical_indicators(df):
    """🔥 Added validation for indicator calculations"""
    # Validate input columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()

    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Volatility Measures
    df['Volatility_5D'] = df['Log_Returns'].rolling(window=5).std()
    df['Volatility_21D'] = df['Log_Returns'].rolling(window=21).std()
    df['Volatility_30D'] = df['Log_Returns'].rolling(window=30).std()
    
    # Momentum Indicators
    df['Momentum_5D'] = df['Close']/df['Close'].shift(5) - 1
    df['Momentum_14D'] = df['Close']/df['Close'].shift(14) - 1
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['20STD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['20STD'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['20STD'] * 2)
    
    # Volume Indicators
    df['Volume_MA5'] = df['Volume'].rolling(5).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df = df.fillna(method='ffill').fillna(0)
    df = df.clip(lower=-1e6, upper=1e6)
    
    # Drop NaN values from rolling calculations
    return df
    

def calculate_features(symbol):
    """🔥 Integrated feature pipeline"""
    data = load_complete_data(symbol)
    df = data['price_data']
    
    # Technical features
    df = calculate_technical_indicators(df)
    
    # News sentiment features
    date_range = pd.date_range(start=START_DATE, end=END_DATE)
    news = get_news_with_dates(symbol, date_range)
    df = add_sentiment_features(df, news)
    
    # Macro features
    df = add_macro_features(df)
    
    # Fundamental features
    fundamentals = data['fundamentals']
    for key, value in fundamentals.items():
        df[key] = value  # Add each fundamental metric as a new column
    
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
            logger.warning(f"Inputed missing column: {col}")
    
    # Final cleaning
    return df.fillna(method='ffill').fillna(0).clip(lower=-1e6, upper=1e6)