import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

# Sklearn for the "black box" model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    # --- Title & Description ---
    st.title("AI-Driven Stock Price Prediction (MVP)")

    st.markdown("""
    **Disclaimer**: This application is for **informational purposes only** and does **not** constitute financial advice. 
    Always do your own research or consult a professional before making investment decisions.

    **Key Features**:
    - Fetches *3 years* of historical data from Yahoo Finance (for training).
    - Displays the last 6 months of *actual/real* prices in a chart **by default**.
    - Predicts *4 horizons*: Next Day, 1 Week, 1 Month, 6 Months (using a single Random Forest model).
    - Minimal chart with real vs. predicted prices.
    """)
    
    # --- Conversational Prompt: Ask Ticker ---
    st.write("Hi! Which stock would you like to check today?")
    ticker = st.text_input("Enter a valid stock ticker (e.g., AAPL, TSLA, AMZN):", value="AAPL")

    # --- Button to Trigger ---
    if st.button("Predict Prices"):
        if not ticker:
            st.error("Please enter a ticker symbol.")
            return
        
        st.write(f"Fetching the last 3 years of data for: **{ticker}**")

        # --- Fetch Data (3 years) ---
        end_date = date.today()
        start_date = end_date - timedelta(days=3*365)  # 3 years ago
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            st.error("No data returned. Check the ticker symbol or try again later.")
            return
        
        # --- Basic Feature Engineering ---
        df_features = feature_engineering(df)
        
        # Define multi-step horizons
        horizon_map = {
            'Next_Day': 1,
            'Next_Week': 5,
            'Next_Month': 21,
            'Next_6Months': 126
        }
        
        # Create columns for each horizon
        for col_name, shift_val in horizon_map.items():
            df_features[col_name] = df_features['Close'].shift(-shift_val)

        # Drop rows that became NaN
        df_features.dropna(inplace=True)
        
        if len(df_features) < 50:
            st.warning("Not enough data left after shifting for predictions. Try a different ticker or date range.")
            return

        # Features & Targets
        X_cols = ['Close', 'MA10', 'MA50', 'RSI', 'MACD']
        y_cols = list(horizon_map.keys())  # Next_Day, Next_Week, etc.

        X = df_features[X_cols]
        y = df_features[y_cols]

        # Train/Test Split (simple, no shuffle)
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        # Scale Features
        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled  = scaler_X.transform(X_test)

        # Build RandomForest Model (multi-output)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)

        # Forward Prediction using the last row of df_features
        last_row_features = df_features.iloc[[-1]][X_cols]
        last_row_scaled   = scaler_X.transform(last_row_features)
        forward_prediction = rf_model.predict(last_row_scaled)[0]  
        # forward_prediction = [pred_NextDay, pred_NextWeek, pred_NextMonth, pred_6Months]
        
        # --- Display Chart (Last 6 Months) ---
        st.write("## Recent 6-Month Chart with Predicted Future Points")
        plot_last_6_months(df_features, forward_prediction, horizon_map)
        
        # --- Display Numeric Predictions ---
        st.write("### Predicted Prices:")
        st.write(f"- **Next Day**: {forward_prediction[0]:.2f} USD")
        st.write(f"- **Next Week**: {forward_prediction[1]:.2f} USD")
        st.write(f"- **Next Month**: {forward_prediction[2]:.2f} USD")
        st.write(f"- **Next 6 Months**: {forward_prediction[3]:.2f} USD")

    else:
        st.info("Enter a ticker and click 'Predict Prices' to proceed.")

def feature_engineering(df):
    """
    Basic indicators: MA(10), MA(50), RSI, MACD.
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
    df['RSI'] = 100 - (100/(1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26

    df.dropna(inplace=True)
    return df

def plot_last_6_months(df_features, forward_prediction, horizon_map):
    """
    Plots the last 6 months of actual closing prices 
    plus the 4 predicted points (Next Day, Next Week, Next Month, 6 Months).
    """
    # Filter data for the last 6 months
    if len(df_features) == 0:
        return  # Safety check

    last_date_in_data = df_features.index[-1]
    six_months_ago = last_date_in_data - pd.DateOffset(months=6)
    df_6mo = df_features[df_features.index >= six_months_ago].copy()
    
    # We'll plot the actual close for these last 6 months
    plt.figure(figsize=(10,5))
    plt.plot(df_6mo.index, df_6mo['Close'], label="Actual Close (Last 6 Mo.)", color='blue')

    # Mark the final known date
    last_close_value = df_6mo['Close'].iloc[-1]
    plt.plot(last_date_in_data, last_close_value, 'ro', label="Latest Known Close")

    # We have predictions for next day, next week, next month, next 6 months
    # We'll place each predicted value on approximate future dates
    horizon_names = list(horizon_map.keys())  # e.g. [Next_Day, Next_Week, Next_Month, Next_6Months]
    for i, horizon in enumerate(horizon_names):
        pred_val = forward_prediction[i]

        # Approx: place each future point some days offset from last_date_in_data
        shift_val = horizon_map[horizon]  # e.g. 1, 5, 21, 126
        future_date = last_date_in_data + pd.Timedelta(days=shift_val)

        plt.plot(future_date, pred_val, 'gx', markersize=10, 
                 label=f"Pred {horizon}")

    plt.title("Last 6 Months Actual Prices + Predicted Future Points")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    st.pyplot()

if __name__ == "__main__":
    main()
