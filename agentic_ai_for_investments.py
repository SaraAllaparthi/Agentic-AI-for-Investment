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

# Hide deprecation warnings for plt
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    # --- Title & Description ---
    st.title("AI-Driven Stock Price Prediction (MVP)")
    st.markdown("""
    **Disclaimer**: This application is for **informational purposes only** and does **not** constitute financial advice. 
    Always do your own research or consult a professional before making investment decisions.
    
    **Key Features**:
    - Fetches *3 years* of historical data from Yahoo Finance.
    - Predicts *4 horizons*: Next Day, 1 Week, 1 Month, 6 Months.
    - Single “black box” Random Forest for multi-step predictions.
    - Minimal chart (optional) + numeric forecasts.
    """)

    # --- Conversational Prompt: Ask Ticker ---
    st.write("Hi! Which stock would you like to check today?")
    ticker = st.text_input("Enter a valid stock ticker (e.g., AAPL, TSLA, AMZN):", value="AAPL")
    
    # --- Button to Trigger ---
    if st.button("Predict Prices"):
        if not ticker:
            st.error("Please enter a ticker symbol.")
            return
        
        st.write("Great! Fetching the last 3 years of data for:", ticker)
        
        # --- Fetch Data (3 years) ---
        end_date = date.today()
        start_date = end_date - timedelta(days=3*365)  # 3 years ago
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            st.error("No data returned. Check the ticker symbol or try again later.")
            return
        
        # --- Basic Feature Engineering ---
        df_features = feature_engineering(df)
        
        # Create multi-step targets: next day, next week (5d), 1 month (~21d), 6 months (~126d)
        horizon_map = {
            'Next_Day': 1,
            'Next_Week': 5,
            'Next_Month': 21,
            'Next_6Months': 126
        }
        
        for col_name, shift_val in horizon_map.items():
            df_features[col_name] = df_features['Close'].shift(-shift_val)
        
        # Drop rows that became NaN after shifting for 6 months horizon
        df_features.dropna(inplace=True)
        
        if len(df_features) < 50:
            st.warning("Not enough data left after shifting for predictions. Try a different ticker or date range.")
            return
        
        # --- Define X & y ---
        # Example features: Close, MA10, MA50, RSI, MACD
        X_cols = ['Close', 'MA10', 'MA50', 'RSI', 'MACD']
        y_cols = list(horizon_map.keys())  # [Next_Day, Next_Week, Next_Month, Next_6Months]
        
        X = df_features[X_cols]
        y = df_features[y_cols]
        
        # --- Train/Test Split (time-based or standard) ---
        # Simple approach: standard train_test_split for MVP
        # For real trading scenario, consider strictly time-based splits.
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        
        # --- Scale Features (optional) ---
        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled  = scaler_X.transform(X_test)
        
        # --- Build RandomForest Model for multi-output ---
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # --- Predict on Test Set (for demonstration) ---
        # We won't display MSE or R2 to keep it simple, as per your request
        predictions_test = rf_model.predict(X_test_scaled)
        # predictions_test is shape (n_samples, 4)
        
        # --- Forward Prediction (for the last available row) ---
        # We'll take the final row from df_features, use today's features, predict future
        last_row = df_features.iloc[[-1]][X_cols]
        last_row_scaled = scaler_X.transform(last_row)
        forward_prediction = rf_model.predict(last_row_scaled)[0]  # [Next_Day, Next_Week, Next_Month, Next_6Months]
        
        # --- Display Numeric Predictions ---
        st.write("### Predictions:")
        st.write(f"- **Next Day**: {forward_prediction[0]:.2f} USD")
        st.write(f"- **Next Week**: {forward_prediction[1]:.2f} USD")
        st.write(f"- **Next Month**: {forward_prediction[2]:.2f} USD")
        st.write(f"- **Next 6 Months**: {forward_prediction[3]:.2f} USD")
        
        # --- Minimal Chart (Optional) ---
        if st.checkbox("Show minimal historical chart"):
            plot_minimal_chart(df_features, forward_prediction, horizon_map)
    
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
    df['RSI'] = 100 - (100/(1+rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26

    df.dropna(inplace=True)
    return df

def plot_minimal_chart(df_features, forward_prediction, horizon_map):
    """
    Plots a minimal line chart of the historical closing price + markers for predictions
    from the final date.
    forward_prediction is [NextDay, NextWeek, NextMonth, Next6Months].
    """
    st.write("### Minimal Historical Chart")
    plt.figure(figsize=(10,5))
    
    # Plot historical close
    plt.plot(df_features.index, df_features['Close'], label="Historical Close", color='blue')
    
    # Mark the last available date
    last_date = df_features.index[-1]
    last_close = df_features['Close'].iloc[-1]
    plt.plot(last_date, last_close, 'ro', label="Latest Known Close")
    
    # Place each predicted horizon as a separate point on the time axis
    # For MVP, we'll just plot them sequentially to the right, not a real future date axis
    # If you want to approximate actual future date, you'd do something like:
    # next_day_date = last_date + BDay(1) in a real scenario. We'll do a naive approach for MVP.
    offset = 1
    horizon_names = list(horizon_map.keys())  # [Next_Day, Next_Week, ...]
    for i, horizon in enumerate(horizon_names):
        predicted_value = forward_prediction[i]
        # We'll shift on x-axis artificially
        future_index = last_date + pd.Timedelta(days=(i+1)*5)  # spaced out for readability
        plt.plot(future_index, predicted_value, 'gx', label=f"Pred {horizon}", markersize=10)
    
    plt.title("Historical Close + Minimal Future Predictions")
    plt.xlabel("Date (approx)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    st.pyplot()

if __name__ == "__main__":
    main()
