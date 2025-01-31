import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import matplotlib.dates as mdates


# Caching the ticker list to speed up the app
@st.cache_data
def load_ticker_list():
    # For demonstration, we use a small curated list.
    # For a comprehensive list, consider loading from a CSV or external source.
    ticker_dict = {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc. (Google)",
        "AMZN": "Amazon.com, Inc.",
        "MSFT": "Microsoft Corporation",
        "TSLA": "Tesla, Inc.",
        "META": "Meta Platforms, Inc. (Facebook)",
        "NFLX": "Netflix, Inc.",
        "NVDA": "NVIDIA Corporation",
        "JPM": "JPMorgan Chase & Co.",
        "V": "Visa Inc.",
        # Add more tickers as needed
    }
    return ticker_dict

# Caching the data fetching to prevent redundant downloads
@st.cache_data
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def feature_engineering(df):
    """
    Add technical indicators: MA10, MA50, RSI, MACD.
    """
    df = df.copy()
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    # Moving Averages
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = (-1 * delta).clip(lower=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26

    df.dropna(inplace=True)
    return df

def create_sequences(data, seq_length):
    """
    Create sequences of data for LSTM.
    """
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def main():
    st.title("📈 Stock Price Prediction - Next 3 Months")

    st.markdown("""
    **Disclaimer**: This application is for **informational purposes only** and does **not** constitute financial advice. 
    Always do your own research or consult a professional before making investment decisions.
    """)

    # 1. Sidebar for Ticker Selection
    st.sidebar.header("Select Stock")
    ticker_dict = load_ticker_list()
    dropdown_options = [f"{name} ({ticker})" for ticker, name in ticker_dict.items()]
    selected_option = st.sidebar.selectbox("Choose a stock:", dropdown_options)
    selected_ticker = selected_option.split("(")[-1].replace(")", "").strip()

    st.write(f"### Selected Stock: **{selected_option}**")

    # 2. Fetch Data
    with st.spinner("Fetching historical data..."):
        end_date = date.today()
        start_date = end_date - timedelta(days=3*365)  # Last 3 years
        data = fetch_data(selected_ticker, start_date, end_date)

    if data.empty:
        st.error("No data found. Please select a different ticker.")
        return

    st.success("Data fetched successfully!")

    # 3. Feature Engineering
    data = feature_engineering(data)

    # 4. Define Target Variable (~3 months ahead)
    horizon_days = 63  # Approx. 3 months
    data['Target'] = data['Close'].shift(-horizon_days)
    data.dropna(inplace=True)  # Remove rows without target

    # Shift features by 1 day to prevent look-ahead bias
    feature_cols = ['Close', 'MA10', 'MA50', 'RSI', 'MACD']
    data[feature_cols] = data[feature_cols].shift(1)
    data.dropna(inplace=True)

    # 5. Train-Test Split (Time-based)
    X = data[feature_cols]
    y = data['Target']

    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if len(X_test) == 0:
        st.warning("Not enough data for testing. Please select a different ticker or adjust the date range.")
        return

    # 6. Feature Scaling
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Create Sequences for LSTM
    seq_length = 60  # Number of previous days to consider
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, seq_length)
    
    # Ensure there is data after sequence creation
    if len(X_test_seq) == 0:
        st.warning("Not enough data after sequence creation. Consider reducing the sequence length.")
        return

    # 8. Build and Train LSTM Model
    st.write("## Training LSTM Model...")

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("Training the LSTM model..."):
        history = model.fit(X_train_seq, y_train_seq, batch_size=32, epochs=20, validation_split=0.1, verbose=0)

    st.success("LSTM model trained successfully!")

    # 9. Make Predictions on Test Set
    st.write("## Making Predictions on Test Set...")

    predictions = model.predict(X_test_seq)
    predictions = predictions.flatten()

    # 10. Inverse Transform to Get Actual Prices
    # Since we scaled the features but not the target, we need to adjust this step
    # Alternatively, you can scale the target as well
    # For simplicity, assume target is in the same scale

    # 11. Plot Actual vs Predicted
    st.write("## Actual vs. Predicted Prices")

    # Align predictions with actual
    plot_dates = X_test.index[seq_length:]
    plot_actual = y_test.iloc[seq_length:].values
    plot_pred = predictions

    plot_df = pd.DataFrame({
        'Date': plot_dates,
        'Actual': plot_actual,
        'Predicted': plot_pred
    })
    plot_df.set_index('Date', inplace=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(plot_df.index, plot_df['Actual'], label='Actual Price', color='blue')
    ax.plot(plot_df.index, plot_df['Predicted'], label='Predicted Price', color='red')
    ax.set_title(f"{selected_ticker} - Actual vs. Predicted Prices on Test Set")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # 12. Predict Future Prices (~3 Months Ahead)
    st.write("## 📈 Future Price Prediction (~3 Months Ahead)")

    last_sequence = X_test_scaled[-seq_length:]
    last_sequence = np.expand_dims(last_sequence, axis=0)  # Shape (1, seq_length, features)
    future_pred = model.predict(last_sequence)
    future_pred = future_pred.flatten()[0]

    # Assuming scaling is applied, inverse transform if necessary
    # Here, since target wasn't scaled, use as is

    # Predict the date ~3 months ahead
    last_date = X_test.index[-1]
    future_date = last_date + timedelta(days=horizon_days)

    st.write(f"**Predicted Closing Price on {future_date.date()}**: **${future_pred:.2f} USD**")

    # 13. Plot Future Prediction on Top of Historical Data
    st.write("## 📊 Historical Prices with Future Prediction")

    # Get the last 6 months of data for reference
    six_months_ago = pd.Timestamp(end_date - timedelta(days=6*30))  # Corrected to pd.Timestamp
    historical_df = data[data.index >= six_months_ago]

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.plot(historical_df.index, historical_df['Close'], label='Historical Close', color='blue')
    ax2.scatter(future_date, future_pred, label='3-Month Prediction', color='green', marker='X', s=100)
    ax2.set_title(f"{selected_ticker} - Historical Prices with 3-Month Prediction")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

if __name__ == "__main__":
    main()
