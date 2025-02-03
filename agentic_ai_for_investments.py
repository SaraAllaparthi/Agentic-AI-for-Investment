# Import the required libraries
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

# --- Helper Function to Predict Next Day Price ---
def predict_next_day(ticker, window_size=60):
    """
    Downloads historical data for the given ticker (last 2 years),
    preprocesses it, loads or trains an LSTM model, and returns:
      - The predicted next day closing price (a float)
      - The historical data DataFrame (for plotting)
    """
    # Download historical data for the last 2 years
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=2 * 365)
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for this ticker. Please check the symbol and try again.")
        st.stop()
    data.reset_index(inplace=True)
    
    # Scale the "Close" prices using a MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    
    # Create sequences for the LSTM model
    X = []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], window_size, 1))
    
    # Use the last sequence to predict the next day price
    last_sequence = scaled_data[-window_size:]
    last_sequence = np.reshape(last_sequence, (1, window_size, 1))
    
    # Check if a pre-trained model exists; if not, train a new one
    model_path = f"models/{ticker}_lstm.h5"
    scaler_path = f"models/{ticker}_scaler.pkl"
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
    else:
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        
        # Create target values
        y = []
        for i in range(window_size, len(scaled_data)):
            y.append(scaled_data[i, 0])
        y = np.array(y)
        
        # Train the model
        early_stop = EarlyStopping(monitor='loss', patience=3)
        model.fit(X, y, epochs=10, batch_size=32, verbose=0, callbacks=[early_stop])
        
        # Save the model and scaler for future use
        if not os.path.exists("models"):
            os.makedirs("models")
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
    
    # Predict the next day price (in scaled space), then inverse-transform it
    pred_scaled = model.predict(last_sequence, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)
    
    return pred[0, 0], data

# --- Streamlit App Setup ---
st.title("Agentic AI for Next Day Closing Price Prediction")
st.caption("Enter a stock ticker to predict the next day's closing price and see the historical trend versus the prediction.")

# Input field for the stock ticker (default is "GOOGL")
ticker = st.text_input("Ticker", value="GOOGL").upper()

if ticker:
    # Get the predicted price and historical data
    predicted_price, hist_data = predict_next_day(ticker)
    
    # Display the predicted price numerically
    st.subheader("Predicted Next Day Closing Price")
    st.write(f"**${predicted_price:.2f}**")
    
    # --- Prepare Data for the Altair Chart ---
    # Historical data: use the Date and Close price
    hist_df = hist_data[['Date', 'Close']].copy()
    hist_df.rename(columns={'Close': 'Price'}, inplace=True)
    hist_df['Date'] = pd.to_datetime(hist_df['Date'])
    hist_df['Type'] = 'Historical'
    
    # Create a prediction trend: a flat line at the predicted price for the next 5 business days
    last_date = hist_df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5, freq='B')
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Price': [predicted_price] * len(future_dates),
        'Type': 'Prediction'
    })
    
    # Combine the historical and prediction DataFrames
    combined_df = pd.concat([hist_df, pred_df], ignore_index=True)
    
    # --- Plot the Altair Chart ---
    chart = alt.Chart(combined_df).mark_line(point=True).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Price:Q', title='Price'),
        color=alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Prediction'], range=['blue', 'red']))
    ).properties(
        width=700,
        height=400,
        title="Historical Prices (Blue) vs. Prediction Trend (Red)"
    )
    
    st.altair_chart(chart, use_container_width=True)
