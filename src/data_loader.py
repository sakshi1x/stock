from venv import logger
import yfinance as yf
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from fredapi import Fred 
from dotenv import load_dotenv
import os

load_dotenv()

FRED_API_KEY=os.getenv('FRED_API_KEY')
NEWS_API_KEY= os.getenv('NEWS_API_KEY')

STOCKS = ["CELH", "CVNA", "UPST", "ALT", "FUBO"]
DATA_PATH = "data/raw"
PERIOD = "5y"


PREDICTION_DAYS = 30
WINDOW_SIZE = 30
START_DATE = os.getenv('START_DATE')
END_DATE = os.getenv('END_DATE')




def load_complete_data(symbol):
    """🔥 Added date alignment and error handling"""
    try:
        # Price data with date filtering
        df = yf.download(symbol, start=START_DATE, end=END_DATE)
        if df.empty:
            raise ValueError(f"No data found for {symbol} in {START_DATE}-{END_DATE}")
            
        # Get date range for alignment
        date_range = pd.date_range(start=START_DATE, end=END_DATE)
        df = df.reindex(date_range).ffill()
        
        # Macro data with date alignment
        fred = Fred(api_key=FRED_API_KEY)
        
        return {
            'price_data': df,
            'news': get_news_with_dates(symbol, date_range),  # 🔥 Fixed news dates
            'fundamentals': get_fundamentals(symbol),
            'macro': {
                'interest_rate': fred.get_series('DFF', START_DATE, END_DATE).ffill(),
                'inflation': fred.get_series('CPIAUCSL', START_DATE, END_DATE)
                            .pct_change(12).ffill()
            }
        }
    except Exception as e:
        print(f"Error loading data for {symbol}: {str(e)}")
        return None

def get_news_with_dates(symbol, date_range):
    """🔥 Align news with proper dates"""
    stock = yf.Ticker(symbol)
    news = stock.news or []
    
    # Create dataframe with news dates
    news_df = pd.DataFrame(index=date_range)
    for article in news:
        pub_date = pd.to_datetime(article.get('providerPublishTime', pd.NaT), unit='s')
        if pd.notna(pub_date):
            news_df.loc[pub_date.floor('D'), 'news'] = article
            
    return news_df.ffill()
def get_fundamentals(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    
    pe_ratio = info.get('trailingPE', np.nan)
    debt_equity = info.get('debtToEquity', np.nan)
    
    # Better handling of missing values
    if np.isnan(pe_ratio):
        pe_ratio = np.nanmedian([info.get('forwardPE', np.nan), 15, 25])  # Sector-based fallback
    
    if np.isnan(debt_equity):
        debt_equity = np.nanmedian([info.get('currentRatio', np.nan), 0.8, 2.5])  # More realistic default

    return {
        'pe_ratio': pe_ratio,
        'debt_equity': debt_equity,
        'ebitda_margin': info.get('ebitdaMargins', np.nan),
        'revenue_growth': info.get('revenueGrowth', np.nan)
    }