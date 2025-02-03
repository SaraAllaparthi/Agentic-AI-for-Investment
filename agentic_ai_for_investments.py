# agentic_ai_for_investments.py

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import os
import joblib
import altair as alt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# Set up page configuration
st.set_page_config(page_title="Agentic AI Dashboard", layout="wide")

# ---------------------------
# Directory for saving models and scalers
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ---------------------------
# Helper Functions

@st.cache_data(show_spinner=True)
def get_stock_data(ticker):
    """Download historical stock data (last 2 years) using yfinance."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=2 * 365)
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

def preprocess_data(data, window_size=60):
    """
    Scale and create sequences for the LSTM model.
    Returns:
      - X: input sequences (shape: [samples, window_size, 1])
      - y: target values
      - scaler: fitted MinMaxScaler (for inverse-transform)
    """
    close_prices = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], window_size, 1))
    return X, y, scaler

def build_model(input_shape):
    """Build and compile an LSTM model."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_and_save_model(ticker, X, y, scaler, epochs=10, batch_size=32):
    """Train the LSTM model and save both the model and scaler."""
    model = build_model((X.shape[1], 1))
    early_stop = EarlyStopping(monitor='loss', patience=3)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    return model

def load_model_and_scaler(ticker):
    """Load the pre-trained model and scaler if available."""
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        return None, None

def predict_next_day(model, last_sequence, scaler):
    """
    Given the model, the last sequence (of shape [window_size, 1]), and the scaler,
    predict the next day's closing price (in original scale).
    """
    input_seq = np.reshape(last_sequence, (1, last_sequence.shape[0], 1))
    pred_scaled = model.predict(input_seq, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)
    return pred[0, 0]

# ---------------------------
# Dashboard Layout

# Sidebar: Input Parameters
st.sidebar.title("Agentic AI Inputs")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="GOOGL").upper()

# Main Page Title
st.title("Agentic AI: Next Day Closing Price Prediction Dashboard")

if ticker:
    st.write(f"Fetching data for **{ticker}**...")
    data = get_stock_data(ticker)
    
    if data.empty:
        st.error("No data found for this ticker. Please check the symbol and try again.")
    else:
        # Preprocess the data (using 2 years of data, 60-day window)
        window_size = 60
        X, y, scaler = preprocess_data(data, window_size)
        
        # Load or train the model
        model, loaded_scaler = load_model_and_scaler(ticker)
        if model is None:
            st.sidebar.info("No pre-trained model found. Training model now...")
            model = train_and_save_model(ticker, X, y, scaler)
            st.sidebar.success("Model trained and saved!")
        else:
            st.sidebar.success("Loaded pre-trained model!")
            scaler = loaded_scaler
        
        # Use the last available 60-day window from historical data for prediction
        last_sequence = scaler.transform(data[['Close']].values)[-window_size:]
        predicted_price = predict_next_day(model, last_sequence, scaler)
        
        # Display the predicted price as a metric
        st.metric(label="Predicted Next Day Closing Price", value=f"${predicted_price:.2f}")
        
        # ---------------------------
        # Prepare data for charting
        
        # Historical DataFrame for plotting
        hist_df = data[['Date', 'Close']].copy()
        hist_df.rename(columns={'Close': 'Price'}, inplace=True)
        hist_df['Date'] = pd.to_datetime(hist_df['Date'])
        hist_df['Type'] = "Historical"
        
        # Create a Prediction Trend DataFrame: a flat line at the predicted price for the next 5 business days
        last_date = hist_df['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5, freq='B')
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Price': [predicted_price] * len(future_dates),
            'Type': "Prediction"
        })
        
        # Combine the historical and prediction data
        combined_df = pd.concat([hist_df, pred_df], ignore_index=True)
        
        # ---------------------------
        # Create an Altair Chart with two lines
        chart = alt.Chart(combined_df).mark_line(point=True).encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Price:Q', title='Price'),
            color=alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Prediction'],
                                                       range=['blue', 'red']))
        ).properties(
            width=700,
            height=400,
            title="Historical Prices vs. Prediction Trend"
        )
        
        st.altair_chart(chart, use_container_width=True)
