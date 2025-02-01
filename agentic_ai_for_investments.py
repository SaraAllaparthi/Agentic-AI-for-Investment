# agentic_ai_stock_prediction.py

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import os
import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# --- Configuration ---
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def get_stock_data(ticker):
    """Fetch historical data for the last 2 years using yfinance."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=2*365)
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

def preprocess_data(data, window_size=60):
    """Scale and create sequences for LSTM."""
    close_prices = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    # Reshape X to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_lstm_model(input_shape):
    """Build and compile the LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_save_model(ticker, X, y, scaler, epochs=20, batch_size=32):
    """Train the LSTM model and save it."""
    model = build_lstm_model((X.shape[1], 1))
    early_stop = EarlyStopping(monitor='loss', patience=3)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=0)
    
    # Save model and scaler
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
    model.save(model_path)
    # Save the scaler parameters (we can use numpy for simplicity)
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.npy")
    np.save(scaler_path, scaler.data_min_)
    np.save(scaler_path.replace("min", "max"), scaler.data_max_)
    
    return model

def load_model_and_scaler(ticker):
    """Load the pre-trained model and scaler if available."""
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.npy")
    scaler_max_path = scaler_path.replace("min", "max")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(scaler_max_path):
        model = load_model(model_path)
        # Reconstruct scaler from saved min and max
        data_min = np.load(scaler_path)
        data_max = np.load(scaler_max_path)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.min_, scaler.scale_ = data_min, (data_max - data_min)
        return model, scaler
    else:
        return None, None

def recursive_forecast(model, last_sequence, n_steps, scaler):
    """Predict future prices for n_steps recursively."""
    predictions = []
    current_seq = last_sequence.copy()
    for _ in range(n_steps):
        # Reshape to (1, window_size, 1)
        pred = model.predict(current_seq.reshape(1, current_seq.shape[0], 1), verbose=0)
        predictions.append(pred[0,0])
        # Append prediction and remove the oldest value
        current_seq = np.append(current_seq[1:], [[pred[0,0]]], axis=0)
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten().tolist()

def get_market_insight(data):
    """A simple insight based on the moving average."""
    # Compute a simple 50-day moving average
    data['MA50'] = data['Close'].rolling(window=50).mean()
    latest_price = data['Close'].iloc[-1]
    latest_ma50 = data['MA50'].iloc[-1]
    if latest_price > latest_ma50:
        return "Market Insight: The stock is trending upward relative to its 50-day average."
    else:
        return "Market Insight: The stock is trending downward relative to its 50-day average."

# --- Streamlit App ---

st.title("Agentic AI to Stock Price Prediction")
st.write("Enter a ticker symbol (as per Yahoo Finance) to see predictions for the next 1 day, 1 week, 1 month, and 6 months.")

ticker = st.text_input("Ticker", value="GOOGL").upper()

if ticker:
    st.write(f"Fetching data for **{ticker}**...")
    data = get_stock_data(ticker)
    
    if data.empty:
        st.error("No data found for this ticker. Please check the symbol and try again.")
    else:
        st.subheader("Historical Stock Price")
        st.line_chart(data.set_index("Date")["Close"])
        
        # Preprocess the data for model training
        window_size = 60
        X, y, scaler = preprocess_data(data, window_size)
        
        # Try to load a pre-trained model; if not found, train one
        model, scaler_loaded = load_model_and_scaler(ticker)
        if model is None:
            st.info("Pre-trained model not found. Training model now (this may take a minute)...")
            model = train_and_save_model(ticker, X, y, scaler)
            st.success("Model trained and saved!")
        else:
            st.success("Loaded pre-trained model.")
            # Use the loaded scaler (if you saved it properly)
            scaler = scaler_loaded
        
        # Get the last sequence from the data to use as seed for predictions
        last_sequence = scaler.transform(data[['Close']].values)[-window_size:]
        
        # Predict future prices
        pred_1_day = recursive_forecast(model, last_sequence, n_steps=1, scaler=scaler)[-1]
        pred_1_week = recursive_forecast(model, last_sequence, n_steps=7, scaler=scaler)[-1]
        pred_1_month = recursive_forecast(model, last_sequence, n_steps=30, scaler=scaler)[-1]
        pred_6_months = recursive_forecast(model, last_sequence, n_steps=180, scaler=scaler)[-1]
        
        st.subheader("Predicted Prices")
        st.write(f"**Next 1 Day:** ${pred_1_day:.2f}")
        st.write(f"**Next 1 Week:** ${pred_1_week:.2f}")
        st.write(f"**Next 1 Month:** ${pred_1_month:.2f}")
        st.write(f"**Next 6 Months:** ${pred_6_months:.2f}")
        
        # Market insight based on current conditions
        insight = get_market_insight(data)
        st.info(insight)
        
        st.markdown("""
        **Disclaimer:** The predictions provided by Agentic AI are for informational purposes only and should not be considered financial advice. Investing in stocks carries risk, and you should do your own research before making any investment decisions.
        """)
