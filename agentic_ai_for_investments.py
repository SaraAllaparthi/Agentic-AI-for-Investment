import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression


def main():
    st.title("Predict Next 3-Month Stock Price")

    ticker_symbol = st.text_input("Enter a stock ticker (e.g. AAPL, TSLA, AMZN):", value="AAPL")

    if st.button("Get 3-Month Prediction"):
        # --- 1) Fetch Data (e.g. last 2 or 3 years) ---
        end_date = date.today()
        start_date = end_date - timedelta(days=3*365)  # 3 years
        df = yf.download(ticker_symbol, start=start_date, end=end_date)

        if df.empty:
            st.error("No data found. Please check the ticker or try again.")
            return

        # --- 2) Feature Engineering ---
        df = df.dropna().copy()
        df.sort_index(inplace=True)

        # Simple indicators
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        
        # RSI
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = (-1*delta).clip(lower=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df['RSI'] = 100 - (100/(1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26

        df.dropna(inplace=True)

        # --- 3) Define Target for ~3 months ahead (63 trading days) ---
        shift_val = 63
        df['Target'] = df['Close'].shift(-shift_val)
        df.dropna(inplace=True)  # remove rows that can't form a target

        # Shift features by 1 day if you want strictly past data
        # (optional—comment out if you don't need it)
        # for col in ['Close','MA10','MA50','RSI','MACD']:
        #     df[col] = df[col].shift(1)
        # df.dropna(inplace=True)

        # --- 4) Train/Test Split (80% / 20%) ---
        X_cols = ['Close','MA10','MA50','RSI','MACD']
        X = df[X_cols]
        y = df['Target']

        split_idx = int(len(df)*0.8)
        train_data = df.iloc[:split_idx]
        test_data  = df.iloc[split_idx:]
        
        X_train = train_data[X_cols]
        y_train = train_data['Target']
        X_test  = test_data[X_cols]
        y_test  = test_data['Target']

        if len(X_test) == 0:
            st.warning("Not enough data for a test set. Try a different ticker or date range.")
            return

        # --- 5) Train Linear Regression ---
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # --- 6) Predict on Test Set ---
        y_test_pred = lr_model.predict(X_test)

        # --- 7) Plot Actual vs Predicted for Test Set ---
        plot_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_test_pred
        }, index=y_test.index)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(plot_df.index, plot_df['Actual'], label='Actual (Test)', color='blue')
        ax.plot(plot_df.index, plot_df['Predicted'], label='Predicted (Test)', color='red')
        ax.set_title(f"{ticker_symbol} - 3-Month Ahead Price (Test Set)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)  # pass fig to avoid the deprecation warning

        # --- 8) Predict the "future" from the last row in the dataset ---
        # We use the final row of features to predict a single point ~3 months ahead
        last_row_features = df.iloc[[-1]][X_cols]
        future_pred_value = lr_model.predict(last_row_features)[0]

        # Display the predicted share price for next 3 months
        st.write(f"### Predicted price ~3 months from the last known date: {future_pred_value:.2f} USD")

    else:
        st.info("Enter a ticker and click 'Get 3-Month Prediction' to proceed.")

if __name__ == "__main__":
    main()
