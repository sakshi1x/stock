import streamlit as st
import pandas as pd
import os
from src.train import train_model
from src.predict import predict_stock
import plotly.graph_objects as go

st.set_page_config(
    page_title="Stock Price Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)
required_dirs = ["models", "data/raw"]
for directory in required_dirs:
    os.makedirs(directory, exist_ok=True)
def load_predictions(symbol):
    """Load predictions from CSV file if it exists"""
    file_path = f"data/raw/{symbol}_predictions.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def create_plot(predictions):
    """Create a Plotly visualization of the predictions"""
    fig = go.Figure()
    
    # Debug: Show available columns
    st.write("Available columns in predictions:", predictions.columns.tolist())
    
    # Try to find likely column names for predictions
    pred_col = None
    date_col = None
    actual_col = None
    
    # Look for prediction column
    possible_pred_cols = ['predicted', 'prediction', 'forecast', 'y_pred']
    for col in possible_pred_cols:
        if col in predictions.columns:
            pred_col = col
            break
    
    # Look for date column
    possible_date_cols = ['date', 'time', 'timestamp', 'index']
    for col in possible_date_cols:
        if col in predictions.columns:
            date_col = col
            break
    
    # Look for actual values column
    possible_actual_cols = ['actual', 'real', 'true', 'y_true']
    for col in possible_actual_cols:
        if col in predictions.columns:
            actual_col = col
            break
    
    # Create plot based on available columns
    if actual_col and date_col:
        fig.add_trace(go.Scatter(
            x=predictions[date_col],
            y=predictions[actual_col],
            name='Actual Price',
            line=dict(color='blue')
        ))
    
    if pred_col and date_col:
        fig.add_trace(go.Scatter(
            x=predictions[date_col],
            y=predictions[pred_col],
            name='Predicted Price',
            line=dict(color='red', dash='dash')
        ))
    else:
        st.error("Could not find prediction data column. Available columns: " + 
                str(predictions.columns.tolist()))
        return None
    
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        height=600
    )
    return fig

def main():
    st.title("Stock Price Prediction Dashboard")
    
    st.sidebar.header("Prediction Settings")
    stocks = ["CELH", "CVNA", "UPST", "ALT", "FUBO"]
    selected_stock = st.sidebar.selectbox("Select Stock Symbol", stocks)
    epochs = st.sidebar.slider("Training Epochs", 50, 200, 100, step=10)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    if st.sidebar.button("Generate Predictions"):
        with st.spinner(f"Training model and generating predictions for {selected_stock}..."):
            try:
                train_model(selected_stock, epochs=epochs)
                predictions = predict_stock(selected_stock)
                predictions.to_csv(f"data/raw/{selected_stock}_predictions.csv")
                st.success(f"Predictions generated for {selected_stock}!")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    st.header(f"Predictions for {selected_stock}")
    predictions = load_predictions(selected_stock)
    
    if predictions is not None:
        st.subheader("Prediction Data")
        st.dataframe(predictions.tail())
        
        st.subheader("Price Visualization")
        fig = create_plot(predictions)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Adjusted statistics section
        pred_col = next((col for col in ['predicted', 'prediction', 'forecast', 'y_pred'] 
                        if col in predictions.columns), None)
        if pred_col:
            st.subheader("Prediction Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Predicted Price", 
                         f"${predictions[pred_col].mean():.2f}")
            with col2:
                st.metric("Prediction Range", 
                         f"${predictions[pred_col].min():.2f} - ${predictions[pred_col].max():.2f}")
    else:
        st.info("No predictions available. Click 'Generate Predictions' to start.")

if __name__ == "__main__":
    main()