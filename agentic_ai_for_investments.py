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
# Page Configuration
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

# Sidebar: Input a single stock ticker
st.sidebar.title("Agentic AI Inputs")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="GOOGL").upper()

st.title("Agentic AI: Stock Trend Dashboard")
st.write("View the historical trend for the last 2 years and the forecast for the next 6 months for the selected stock.")

if ticker:
    st.write(f"Fetching data for **{ticker}**...")
    data = get_stock_data(ticker)
    if data.empty:
        st.error("No data found for the ticker. Please check the symbol.")
    else:
        # Preprocess the data using a 60-day window
        window_size = 60
        X, y, scaler = preprocess_data(data, window_size)
        
        # Load or train the model
        model, loaded_scaler = load_model_and_scaler(ticker)
        if model is None:
            st.info("No pre-trained model found. Training model now...")
            model = train_and_save_model(ticker, X, y, scaler)
            st.success("Model trained and saved!")
        else:
            st.success("Loaded pre-trained model!")
            scaler = loaded_scaler
        
        # Use the last available 60-day window for prediction
        last_sequence = scaler.transform(data[['Close']].values)[-window_size:]
        next_day_pred = predict_next_day(model, last_sequence, scaler)
        st.metric("Predicted Next Day Closing Price", f"${next_day_pred:.2f}")
        
        # Historical DataFrame for the last 2 years
        hist_df = data.copy()
        hist_df['Date'] = pd.to_datetime(hist_df['Date'])
        hist_df = hist_df[['Date', 'Close']].copy()
        hist_df.rename(columns={'Close': 'Price'}, inplace=True)
        
        # Forecast DataFrame for the next 6 months (approx. 126 business days)
        n_forecast = 126
        future_preds = recursive_forecast(model, last_sequence, n_steps=n_forecast, scaler=scaler)
        last_hist_date = hist_df['Date'].max()
        future_dates = pd.date_range(start=last_hist_date + pd.Timedelta(days=1), periods=n_forecast, freq='B')
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Price': future_preds
        })
        
        # Define overall x-axis domain from the earliest historical date to the last predicted date
        overall_domain = [hist_df['Date'].min(), pred_df['Date'].max()]
        
        # Create an Altair chart for historical data (solid blue line)
        chart_hist = alt.Chart(hist_df).mark_line(color='blue').encode(
            x=alt.X('Date:T', title='Date', scale=alt.Scale(domain=overall_domain)),
            y=alt.Y('Price:Q', title='Price')
        )
        
        # Create an Altair chart for forecast data (dashed red line)
        chart_pred = alt.Chart(pred_df).mark_line(color='red', strokeDash=[5,5]).encode(
            x=alt.X('Date:T', title='Date', scale=alt.Scale(domain=overall_domain)),
            y=alt.Y('Price:Q', title='Price')
        )
        
        # Add a text annotation for the forecast region
        forecast_mid_date = pred_df['Date'].iloc[len(pred_df)//2]
        forecast_mid_price = pred_df['Price'].mean()
        forecast_annotation = alt.Chart(pd.DataFrame({
            'Date': [forecast_mid_date],
            'Price': [forecast_mid_price]
        })).mark_text(
            align='center',
            baseline='middle',
            dy=-10,
            color='red',
            fontSize=12
        ).encode(
            x='Date:T',
            y='Price:Q',
            text=alt.value("Forecast")
        )
        
        # Layer the historical, forecast, and annotation charts together
        chart = alt.layer(chart_hist, chart_pred, forecast_annotation).properties(
            width=700,
            height=400,
            title=f"{ticker}: Historical vs. 6-Month Forecast"
        )
        
        st.altair_chart(chart, use_container_width=True)
