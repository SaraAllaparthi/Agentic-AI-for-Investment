import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

def main():
    st.title("Enhanced Stock Price Prediction (MVP)")

    st.markdown("""
    **Key Features**:
    - Trains & tests on *3 years* of historical data (from Yahoo Finance).
    - Uses **4 separate horizons**: Next Day (1 trading day), Next Week (5), Next Month (~21), Next 6 Months (~126).
    - For each horizon, we build an **ensemble** of 3 models (Linear, Random Forest, XGBoost) and **average** predictions.
    - We **plot** Actual vs. Predicted on the **test set** for each horizon to show backtest performance.
    - We also provide a **future** (post-last-date) prediction for each horizon as a numeric value.
    
    **Disclaimer**: This application is for **informational purposes only** and does **not** constitute financial advice. 
    Predictions can be inaccurate; always do your own research or consult a professional before making investment decisions.
    """)

    ticker = st.text_input("Enter a valid stock ticker (e.g. AAPL, TSLA, AMZN):", value="AAPL")

    if st.button("Predict Prices"):
        if not ticker:
            st.error("Please enter a ticker symbol.")
            return
        
        st.write(f"Fetching data for {ticker}...")

        # 3 years of data
        end_date = date.today()
        start_date = end_date - timedelta(days=3*365)
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            st.error("No data returned. Check the ticker symbol or try again later.")
            return

        # Feature Engineering
        df_feat = feature_engineering(df)

        # We define 4 horizons with shift in trading days
        horizon_map = {
            'Next_Day': 1,
            'Next_Week': 5,
            'Next_Month': 21,
            'Next_6Months': 126
        }

        # Create new columns for each horizon
        for horizon_name, shift_val in horizon_map.items():
            df_feat[horizon_name] = df_feat['Close'].shift(-shift_val)

        # Drop rows that became NaN due to shifting
        df_feat.dropna(inplace=True)

        if len(df_feat) < 50:
            st.warning("Not enough data left after shifting. Try a different ticker or date range.")
            return

        # We will use these columns as features
        X_cols = ['Close', 'MA10', 'MA50', 'RSI', 'MACD']
        
        results_df = df_feat.copy()  # We'll store predictions for each horizon in this

        # Time-based train/test split (e.g., first 80% train, last 20% test)
        train_size = int(len(df_feat)*0.8)
        train_data = df_feat.iloc[:train_size]
        test_data  = df_feat.iloc[train_size:]

        # For plotting
        test_index = test_data.index

        # We'll store predictions in these columns for plotting
        for horizon_name in horizon_map.keys():
            results_df[f"Pred_{horizon_name}"] = np.nan

        # For each horizon, build an ensemble model, predict on test set, store predictions
        st.write("## Training & Predicting on Each Horizon...")
        
        # We'll do a final forward prediction from the last row in df_feat
        last_row_features = df_feat.iloc[[-1]][X_cols]
        
        # Scale features (recommended). We'll do a minimal approach: same scaler across train/test
        # If you want separate scalers per horizon, you can, but it's usually the same
        scaler = MinMaxScaler()

        # Fit on entire dataset's feature range (some might prefer fit only on train)
        # For a more realistic scenario, fit only on train_data, then transform test_data, etc.
        scaler.fit(df_feat[X_cols])
        X_train_scaled = scaler.transform(train_data[X_cols])
        X_test_scaled  = scaler.transform(test_data[X_cols])
        X_last_scaled  = scaler.transform(last_row_features[X_cols])

        future_predictions = {}  # to store numeric future preds

        for horizon_name, shift_val in horizon_map.items():
            st.write(f"### Horizon: {horizon_name}")
            
            # y is the shifted close
            y_train = train_data[horizon_name]
            y_test  = test_data[horizon_name]
            
            # Subset
            # For a robust approach, do: X_train = scaler.transform(train_data[X_cols]) etc.
            X_train = X_train_scaled[:len(y_train)]  # same number of rows as train_data
            X_test  = X_test_scaled[len(y_train):]   # correspond to test_data rows

            # Build an ensemble of 3 regressors
            lr_model  = LinearRegression()
            rf_model  = RandomForestRegressor(n_estimators=100, random_state=42)
            xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            
            # Fit each on training data
            lr_model.fit(X_train, y_train)
            rf_model.fit(X_train, y_train)
            xgb_model.fit(X_train, y_train)
            
            # Predict on test data
            lr_pred  = lr_model.predict(X_test)
            rf_pred  = rf_model.predict(X_test)
            xgb_pred = xgb_model.predict(X_test)
            
            # Average ensemble
            ensemble_pred = (lr_pred + rf_pred + xgb_pred) / 3.0
            
            # Store these predictions in results_df
            results_df.loc[test_index, f"Pred_{horizon_name}"] = ensemble_pred

            # Plot Actual vs. Predicted for this horizon (test set only)
            plot_test_predictions(horizon_name, test_data[horizon_name], ensemble_pred, test_index)
            
            # Future (post-last-date) prediction using the final row's features
            # We'll do the same ensemble approach
            lr_future  = lr_model.predict(X_last_scaled)
            rf_future  = rf_model.predict(X_last_scaled)
            xgb_future = xgb_model.predict(X_last_scaled)
            final_ensemble_future = (lr_future + rf_future + xgb_future) / 3.0
            
            future_predictions[horizon_name] = final_ensemble_future[0]

        # --- Show final numeric predictions for the future ---
        st.write("## Future Predictions (After Last Known Date)")
        for horizon_name in horizon_map.keys():
            st.write(f"- **{horizon_name}**: {future_predictions[horizon_name]:.2f} USD")

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

def plot_test_predictions(horizon_name, actual_series, predicted_array, test_index):
    """
    Plots a line chart of Actual vs. Predicted for the test set
    for the given horizon.
    actual_series is from test_data[horizon_name].
    predicted_array is ensemble predictions (same length).
    test_index is the index corresponding to test_data rows.
    """
    st.write(f"**Test Set: Actual vs. Predicted for {horizon_name}**")
    fig, ax = plt.subplots(figsize=(10,4))
    
    # Convert to numeric for plotting
    actual_values = actual_series.values
    ax.plot(test_index, actual_values, label="Actual", color='blue')
    
    ax.plot(test_index, predicted_array, label="Predicted (Ensemble)", color='red')
    
    ax.set_title(f"{horizon_name} - Actual vs. Ensemble Predicted (Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
