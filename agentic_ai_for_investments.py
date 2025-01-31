import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

def main():
    st.title("3-Month Ahead Stock Price - Ensemble Prediction")

    # Text input for ticker
    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA):", value="AAPL")

    if st.button("Predict 3-Month Price"):
        # 1) Download last 3 years of data
        end_date = date.today()
        start_date = end_date - timedelta(days=3*365)
        df = yf.download(ticker_symbol, start=start_date, end=end_date)

        if df.empty:
            st.error("No data found. Please try a different ticker or date range.")
            return

        # 2) Feature Engineering
        df = df.dropna().copy()
        df.sort_index(inplace=True)

        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA50'] = df['Close'].rolling(50).mean()

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

        # 3) Define Target for ~3 months (~63 trading days)
        horizon_days = 63
        df['Target'] = df['Close'].shift(-horizon_days)
        df.dropna(inplace=True)  # Remove rows that don't have a target

        # 4) Select features & time-based split
        X_cols = ['Close', 'MA10', 'MA50', 'RSI', 'MACD']
        X = df[X_cols]
        y = df['Target']

        split_idx = int(len(df) * 0.8)
        train_data = df.iloc[:split_idx]
        test_data  = df.iloc[split_idx:]

        X_train = train_data[X_cols]
        y_train = train_data['Target']
        X_test  = test_data[X_cols]
        y_test  = test_data['Target']

        if len(X_test) == 0:
            st.warning("Not enough data for a test set. Try a different ticker or date range.")
            return

        # 5) Scale the features
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # 6) Build 3 models & average predictions
        #    (You can tune hyperparameters more if desired)
        lr_model = LinearRegression()
        rdg_model = Ridge(alpha=1.0)
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)  # smaller model for speed

        lr_model.fit(X_train_scaled, y_train)
        rdg_model.fit(X_train_scaled, y_train)
        rf_model.fit(X_train_scaled, y_train)

        lr_pred_test = lr_model.predict(X_test_scaled)
        rdg_pred_test = rdg_model.predict(X_test_scaled)
        rf_pred_test = rf_model.predict(X_test_scaled)

        # Ensemble test predictions
        ensemble_test_pred = (lr_pred_test + rdg_pred_test + rf_pred_test) / 3.0

        # 7) Plot Actual vs. Predicted on Test
        plot_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': ensemble_test_pred
        }, index=y_test.index)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(plot_df.index, plot_df['Actual'], label='Actual (Test)', color='blue')
        ax.plot(plot_df.index, plot_df['Predicted'], label='Predicted (Test)', color='red')
        ax.set_title(f"{ticker_symbol} - ~3 Month Ahead Ensemble Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # 8) Single Future Prediction from last row
        last_feats = df.iloc[[-1]][X_cols]
        last_scaled = scaler.transform(last_feats)
        lr_future  = lr_model.predict(last_scaled)
        rdg_future = rdg_model.predict(last_scaled)
        rf_future  = rf_model.predict(last_scaled)
        ensemble_future = (lr_future + rdg_future + rf_future) / 3.0
        future_val = ensemble_future[0]

        st.write(f"**Predicted Price ~3 Months After Last Date**: {future_val:.2f} USD")

    else:
        st.info("Enter a ticker and click 'Predict 3-Month Price' to run the model.")

if __name__ == "__main__":
    main()
