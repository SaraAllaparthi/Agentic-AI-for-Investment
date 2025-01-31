import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.dates as mdates


def main():
    st.title("Next-Day Stock Price Prediction (Linear, Ridge, Random Forest)")

    # Ticker input
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", value="AAPL")

    if st.button("Predict Next-Day Price"):
        # 1) Fetch Data (last 3 years)
        end_date = date.today()
        start_date = end_date - timedelta(days=3*365)
        df = yf.download(ticker, start=start_date, end=end_date)

        if df.empty:
            st.error("No data found. Please try another ticker.")
            return

        # 2) Feature Engineering
        df = df.dropna().copy()
        df.sort_index(inplace=True)

        # Example indicators
        df['MA10'] = df['Close'].rolling(window=10).mean()

        # RSI
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = (-1*delta).clip(lower=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26

        df.dropna(inplace=True)

        # 3) Define Target as Next-Day Close
        df['Target'] = df['Close'].shift(-1)

        # Shift features by 1 day so we only use "yesterday's" data
        feature_cols = ['Close', 'MA10', 'RSI', 'MACD']  # or add 'MA50' if you like
        for col in feature_cols:
            df[col] = df[col].shift(1)
        df.dropna(inplace=True)

        # 4) Split Data (time-based 80% / 20%)
        # We'll avoid using shuffle=True because it's time series
        X = df[feature_cols]
        y = df['Target']

        split_idx = int(len(df)*0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if len(X_test) == 0:
            st.warning("Not enough data for a test set. Adjust your date range.")
            return

        # 5) Train Models
        # a) Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_predictions = lr_model.predict(X_test)

        # b) Ridge
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        ridge_predictions = ridge_model.predict(X_test)

        # c) Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)

        # 6) Plot Results: Actual vs. Predictions
        plot_df = pd.DataFrame({
            'Date': X_test.index,
            'Actual Price': y_test.values,
            'Linear Regression Prediction': lr_predictions,
            'Ridge Regression Prediction': ridge_predictions,
            'Random Forest Prediction': rf_predictions
        })
        plot_df.set_index('Date', inplace=True)

        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot actual
        ax.plot(plot_df.index, plot_df['Actual Price'],
                label='Actual Price', color='blue')
        # LR
        ax.plot(plot_df.index, plot_df['Linear Regression Prediction'],
                label='Linear Regression', color='red')
        # Ridge
        ax.plot(plot_df.index, plot_df['Ridge Regression Prediction'],
                label='Ridge Regression', color='orange')
        # RF
        ax.plot(plot_df.index, plot_df['Random Forest Prediction'],
                label='Random Forest', color='green')

        ax.set_title(f"{ticker} - Next-Day Closing Price Predictions")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()

        # Improve date formatting
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

    else:
        st.info("Enter a ticker and click 'Predict Next-Day Price' to start.")

if __name__ == "__main__":
    main()
