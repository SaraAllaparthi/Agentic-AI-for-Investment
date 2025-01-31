import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta

# Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Multi-Horizon Stock Price Prediction")

    st.markdown("""
    This Streamlit app fetches historical stock data from Yahoo Finance, 
    engineers a few technical indicators, and then trains **4 separate 
    Linear Regression models** to predict the closing price for:
    - Next Day (1 trading day ahead)
    - Next Week (5 trading days ahead)
    - Next Month (~21 trading days ahead)
    - Next 6 Months (~126 trading days ahead)
    """)

    # ----- SIDEBAR CONTROLS ----- #
    st.sidebar.header("Model Configuration")

    # 1) Ticker Selection
    ticker_options = ["AAPL", "GOOG", "TSLA", "AMZN", "MSFT", "Custom Ticker"]
    chosen_ticker = st.sidebar.selectbox("Select a Ticker", ticker_options, index=0)
    
    custom_ticker = None
    if chosen_ticker == "Custom Ticker":
        custom_ticker = st.sidebar.text_input("Enter a valid Yahoo Finance ticker:", "IBM")
        ticker = custom_ticker
    else:
        ticker = chosen_ticker

    # 2) Start Date (no End Date; we fetch up to 'today')
    default_start = date.today() - timedelta(days=365*5)  # 5 years ago
    start_date = st.sidebar.date_input("Start Date (for data & training)", value=default_start)
    
    if st.sidebar.button("Run Prediction"):
        if not ticker:
            st.error("Please enter a valid ticker symbol.")
            return
        
        with st.spinner("Fetching data from Yahoo Finance..."):
            df = yf.download(ticker, start=start_date, end=date.today())
        
        if df.empty or len(df) < 150:
            st.warning("Not enough data returned. Try a different ticker or earlier start date.")
            return
        
        # ----- DATA PREPARATION ----- #
        df = df.dropna()
        df = feature_engineering(df)
        
        # Create multi-horizon target columns
        horizon_shifts = {
            'Next_Day': 1,
            'Next_Week': 5,
            'Next_Month': 21,
            'Next_6Months': 126
        }
        
        for horizon_name, shift_val in horizon_shifts.items():
            df[horizon_name] = df['Close'].shift(-shift_val)
        
        # Drop final rows that are now NaN due to shifting
        # We'll lose up to 126 rows at the end for Next_6Months
        df.dropna(inplace=True)
        
        # If there's not much data left, we might skip
        if len(df) < 200:
            st.warning("After shifting for 6 months horizon, not enough data to train. Try earlier start date.")
            return

        # Show the final DataFrame structure
        st.subheader(f"Data after Feature Engineering & Shifting for {ticker}")
        st.dataframe(df.head(10))

        # ----- TRAIN/TEST SPLIT ----- #
        # We'll do a time-based split: first 80% is train, last 20% is test
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df  = df.iloc[train_size:]

        # We'll define the feature columns:
        features = ['Close', 'MA10', 'MA50', 'RSI', 'MACD']  # You can remove MA50 if needed

        results = {}
        predictions_df = test_df[['Close']].copy()
        
        # ----- TRAIN/TEST for Each Horizon ----- #
        for horizon_name in horizon_shifts.keys():
            # Our target is the horizon column
            y_train = train_df[horizon_name]
            y_test  = test_df[horizon_name]
            
            X_train = train_df[features]
            X_test  = test_df[features]
            
            # Simple Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predict on test set
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2  = r2_score(y_test, y_pred)
            
            results[horizon_name] = (mse, r2)
            predictions_df[f"Pred_{horizon_name}"] = y_pred
        
        # ----- SHOW EVALUATION METRICS ----- #
        st.subheader("Model Performance (Test Set)")
        metric_table = []
        for horizon_name, (mse, r2) in results.items():
            metric_table.append([horizon_name, mse, r2])
        
        metrics_df = pd.DataFrame(metric_table, columns=["Horizon", "MSE", "R²"])
        st.dataframe(metrics_df)

        # ----- PLOT: Actual vs. Predicted (TEST SET) ----- #
        st.subheader("Actual vs. Predicted (Test Set)")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_df.index, test_df['Close'], label='Actual Close', color='blue')
        
        # We'll plot the predictions for each horizon
        colors = ['red', 'green', 'orange', 'purple']
        for (horizon_name, c) in zip(horizon_shifts.keys(), colors):
            ax.plot(predictions_df.index, predictions_df[f"Pred_{horizon_name}"], 
                    label=f"Prediction {horizon_name}", color=c, alpha=0.7)
        
        ax.set_title(f"{ticker} - Actual vs. Multi-Horizon Predictions (Test Set)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # ----- FORWARD PREDICTIONS FROM MOST RECENT DAY ----- #
        st.subheader("Forward Predictions: Next Day, Next Week, Next Month, Next 6 Months")
        # We'll take the last row of the full df (before we dropped NA) 
        # Actually let's take the last row from df (the final row used in training?).
        last_row = df.iloc[[-1]].copy()

        forward_preds = {}
        # We'll just re-use the same 4 trained models from the code above or re-train
        # but for a simpler approach, let's do a fresh re-train on the full df for each horizon,
        # then predict the future row (the final row might not have a "horizon" value).
        # This approach ensures we use all data for training.
        
        for horizon_name, shift_val in horizon_shifts.items():
            # Full training 
            # Drop last 'shift_val' rows from df to avoid target leakage
            # so features are aligned properly for the horizon
            max_index_for_training = len(df) - shift_val
            df_for_training = df.iloc[:max_index_for_training].copy()
            
            X_all = df_for_training[features]
            y_all = df_for_training[horizon_name]
            
            # Train new model
            model = LinearRegression()
            model.fit(X_all, y_all)
            
            # Predict using the final available row of features
            # We must ensure that final row of features is valid (the last row might 
            # or might not be the same as df_for_training's last row).
            # We'll use the final row from df (which might represent today's data).
            X_future = last_row[features]
            
            pred_value = model.predict(X_future)[0]
            forward_preds[horizon_name] = pred_value
        
        # Display the forward predictions
        for horizon_name, val in forward_preds.items():
            st.write(f"**{horizon_name} Prediction:** {val:.2f} USD")

        # ----- HISTORICAL CHART (entire dataset) ----- #
        st.subheader("Historical Price Chart (Last 5+ Years)")
        fig2, ax2 = plt.subplots(figsize=(12,6))
        ax2.plot(df.index, df['Close'], label='Close Price', color='blue')
        ax2.set_title(f"{ticker} - Historical Closing Price")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price (USD)")
        ax2.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    else:
        st.info("Configure parameters in the sidebar and click 'Run Prediction' to start.")


def feature_engineering(df):
    """
    Basic feature engineering: 
      - Add MA(10), MA(50), RSI, MACD
    """
    # Sort by date just in case
    df = df.sort_index()
    # Calculate MAs
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
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


if __name__ == "__main__":
    main()
