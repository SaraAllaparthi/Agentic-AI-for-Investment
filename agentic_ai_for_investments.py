import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta

# Sklearn libraries for Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def main():
    st.title("Linear Regression - Next Day Stock Price Prediction")

    st.markdown("""
    **Goal**: Predict the **next day’s closing price** using a **Linear Regression** model.

    **Key Steps**:
    - Fetch last 3 years of data from Yahoo Finance.
    - Create technical indicators (MA10, MA50, RSI, MACD).
    - Define target as *next day’s close*.
    - Train/test split (80% train, 20% test).
    - Evaluate & visualize predictions vs. actual values on the test set.
    
    **Disclaimer**: This is for **informational purposes only**, not financial advice.
    """)

    # 1) Sidebar Ticker Input
    st.sidebar.header("Select Stock Ticker")
    ticker_symbol = st.sidebar.text_input("Enter a valid ticker (e.g. AAPL, TSLA):", value="AAPL")

    if st.sidebar.button("Run Prediction"):
        if not ticker_symbol:
            st.error("Please enter a ticker symbol.")
            return

        # 2) Download Data
        with st.spinner("Fetching data from Yahoo Finance..."):
            end_date = date.today()
            start_date = end_date - timedelta(days=3*365)  # last 3 years
            df = yf.download(ticker_symbol, start=start_date, end=end_date)

        if df.empty:
            st.error("No data found. Check the ticker symbol or try later.")
            return

        # 3) Feature Engineering
        df = feature_engineering(df)

        # 4) Define Target (Next Day Close)
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        # Shift features by 1 day to ensure only past data is used
        shifted_features = ['Close', 'MA10', 'MA50', 'RSI', 'MACD']
        for col in shifted_features:
            df[col] = df[col].shift(1)
        df.dropna(inplace=True)

        # 5) Split Data
        features = ['Close', 'MA10', 'MA50', 'RSI', 'MACD']
        X = df[features]
        y = df['Target']

        train_size = int(len(df)*0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        if len(X_test) == 0:
            st.warning("Not enough data for a test set. Try a different ticker or more data.")
            return

        # Optional: scale data (comment out if not needed)
        # scaler = MinMaxScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)

        # 6) Train Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # Predict on test set
        lr_predictions = lr_model.predict(X_test)

        # Evaluate
        lr_mse = mean_squared_error(y_test, lr_predictions)
        lr_r2  = r2_score(y_test, lr_predictions)

        # 7) Display Results
        st.subheader("Model Evaluation")
        st.write(f"**Test MSE**: {lr_mse:.2f}")
        st.write(f"**Test R²**: {lr_r2:.4f}")

        # 8) Plot Actual vs. Predicted
        st.subheader("Test Set: Actual vs. Predicted")
        plot_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': lr_predictions
        }, index=y_test.index)

        plt.figure(figsize=(12, 5))
        plt.plot(plot_df.index, plot_df['Actual'], label='Actual', color='blue')
        plt.plot(plot_df.index, plot_df['Predicted'], label='Predicted', color='red')
        plt.title(f"{ticker_symbol} - Next Day Close: Actual vs. Predicted")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.xticks(rotation=45)
        st.pyplot()

    else:
        st.info("Configure a ticker in the sidebar and click 'Run Prediction' to start.")

def feature_engineering(df):
    """
    Add technical indicators: MA10, MA50, RSI, MACD.
    Drop rows with NA afterwards.
    """
    df = df.copy()
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    # MA10, MA50
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100/(1+rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26

    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    main()
