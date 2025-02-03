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
# Page configuration
st.set_page_config(page_title="Agentic AI Stock Trend Dashboard", layout="wide")

# ---------------------------
# Directory for saving models and scalers
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ---------------------------
# Helper Functions

@st.cache_data(show_spinner=True)
def get_stock_data(ticker):
    """Download historical data for the last 2 years using yfinance."""
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
      - scaler: fitted MinMaxScaler for inverse-transform
    """
    close_prices = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
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
    Given the model and the last sequence (shape: [window_size, 1]),
    predict the next day's closing price (in original scale).
    """
    input_seq = last_sequence.reshape(1, last_sequence.shape[0], 1)
    pred_scaled = model.predict(input_seq, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)
    return pred[0, 0]

def recursive_forecast(model, last_sequence, n_steps, scaler):
    """
    Predict the next n_steps closing prices recursively.
    Returns a list of predictions.
    """
    predictions = []
    current_seq = last_sequence.copy()  # shape: (window_size, 1)
    for _ in range(n_steps):
        input_seq = current_seq.reshape(1, current_seq.shape[0], 1)
        pred_scaled = model.predict(input_seq, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0, 0]
        predictions.append(pred)
        new_val_scaled = pred_scaled[0, 0]
        current_seq = np.append(current_seq[1:], [[new_val_scaled]], axis=0)
    return predictions

# ---------------------------
# Dashboard Layout

# Sidebar: Allow the user to enter up to 3 share tickers (comma-separated)
tickers_input = st.sidebar.text_input("Enter up to 3 share tickers (comma-separated)", 
                                        value="GOOGL, AAPL, MSFT")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][:3]

st.title("Agentic AI: Stock Trend Dashboard")
st.write("Select up to three shares to view their trends for the last 2 years and a forecast for the next 6 months using our ML model.")

# Process each ticker
for ticker in tickers:
    st.subheader(f"Stock: {ticker}")
    
    # Download historical data (last 2 years)
    data = get_stock_data(ticker)
    if data.empty:
        st.error(f"No data found for {ticker}.")
        continue
    
    # Preprocess the data (using a 60-day window)
    window_size = 60
    X, y, scaler = preprocess_data(data, window_size)
    
    # Load or train the model for this ticker
    model, loaded_scaler = load_model_and_scaler(ticker)
    if model is None:
        st.info(f"No pre-trained model for {ticker}. Training model now...")
        model = train_and_save_model(ticker, X, y, scaler)
        st.success(f"Model for {ticker} trained and saved!")
    else:
        st.success(f"Loaded pre-trained model for {ticker}.")
        scaler = loaded_scaler
    
    # Use the last available 60-day window for prediction
    last_sequence = scaler.transform(data[['Close']].values)[-window_size:]
    
    # Forecast next 6 months (approx. 126 business days) using recursive forecasting
    n_forecast = 126
    future_preds = recursive_forecast(model, last_sequence, n_steps=n_forecast, scaler=scaler)
    
    # Historical DataFrame for the last 2 years
    hist_df = data.copy()
    hist_df['Date'] = pd.to_datetime(hist_df['Date'])
    hist_df = hist_df[['Date', 'Close']].copy()
    hist_df.rename(columns={'Close': 'Price'}, inplace=True)
    
    # Forecast DataFrame for the next 6 months
    last_hist_date = hist_df['Date'].max()
    future_dates = pd.date_range(start=last_hist_date + pd.Timedelta(days=1), periods=n_forecast, freq='B')
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Price': future_preds
    })
    
    # Determine overall x-axis domain: from the earliest historical date to the last predicted date
    overall_domain = [hist_df['Date'].min(), pred_df['Date'].max()]
    
    # Create Altair chart for historical data (blue line)
    chart_hist = alt.Chart(hist_df).mark_line(color='blue').encode(
        x=alt.X('Date:T', title='Date', scale=alt.Scale(domain=overall_domain)),
        y=alt.Y('Price:Q', title='Price')
    )
    
    # Create Altair chart for forecast data (red line)
    chart_pred = alt.Chart(pred_df).mark_line(color='red').encode(
        x=alt.X('Date:T', title='Date', scale=alt.Scale(domain=overall_domain)),
        y=alt.Y('Price:Q', title='Price')
    )
    
    # Create a vertical dashed rule at the last historical date
    vertical_rule = alt.Chart(pd.DataFrame({'Date': [last_hist_date]})).mark_rule(
        color='black', strokeDash=[5, 5]
    ).encode(
        x=alt.X('Date:T')
    )
    
    # Layer the charts together
    chart = alt.layer(chart_hist, chart_pred, vertical_rule).properties(
        width=700,
        height=400,
        title=f"{ticker}: Historical (Blue) vs. 6-Month Forecast (Red)"
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Explanation for the vertical rule
    st.markdown(
        f"""<div style="font-size:14px; margin-bottom:20px;">
            <b>Note:</b> The vertical dashed line at {last_hist_date.strftime('%Y-%m-%d')} indicates 
            the transition from historical data to the forecast period.
            </div>""",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
