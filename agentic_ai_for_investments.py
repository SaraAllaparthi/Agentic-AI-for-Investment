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
    st.title("Faster Multi-Horizon Prediction (Reduced Data & Estimators)")
    
    st.markdown("""
    **Changes** for speed:
    - **2 years** of data instead of 3.
    - **20 trees** (n_estimators=20) in RandomForest and XGBoost (down from 100).
    - 4 horizons, 3-model ensemble, but lighter training.

    **Disclaimer**: For informational purposes only, not financial advice.
    """)

    ticker = st.text_input("Enter a stock ticker:", value="AAPL")
    if st.button("Predict"):
        if not ticker:
            st.error("Please enter a ticker.")
            return
        
        st.write(f"Fetching last 2 years of data for {ticker}...")
        end_date = date.today()
        start_date = end_date - timedelta(days=2*365)  # 2 years
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            st.error("No data. Try another ticker or adjust date range.")
            return
        
        df_feat = feature_engineering(df)
        
        # Horizons
        horizon_map = {
            'Next_Day': 1,
            'Next_Week': 5,
            'Next_Month': 21,
            'Next_6Months': 126
        }
        
        for hname, shift_val in horizon_map.items():
            df_feat[hname] = df_feat['Close'].shift(-shift_val)
        df_feat.dropna(inplace=True)
        
        if len(df_feat) < 50:
            st.warning("Not enough rows left after shifting.")
            return
        
        X_cols = ['Close','MA10','MA50','RSI','MACD']
        
        # Time-based split (80% train, 20% test)
        train_size = int(len(df_feat)*0.8)
        train_data = df_feat.iloc[:train_size]
        test_data  = df_feat.iloc[train_size:]
        
        scaler = MinMaxScaler()
        scaler.fit(df_feat[X_cols])
        
        X_train_scaled = scaler.transform(train_data[X_cols])
        X_test_scaled  = scaler.transform(test_data[X_cols])
        
        test_index = test_data.index
        
        # We'll store predictions in the dataframe for plotting
        for hname in horizon_map:
            df_feat[f"Pred_{hname}"] = np.nan
        
        # For final forward prediction
        last_row_features = df_feat.iloc[[-1]][X_cols]
        X_last_scaled = scaler.transform(last_row_features)
        
        future_preds = {}
        
        for hname, shift_val in horizon_map.items():
            st.write(f"### Horizon: {hname}")
            
            y_train = train_data[hname]
            y_test  = test_data[hname]
            
            # Subsets for train/test
            X_train = X_train_scaled[:len(y_train)]
            X_test  = X_test_scaled[len(y_train):]
            
            # Light ensemble
            lr  = LinearRegression()
            rf  = RandomForestRegressor(n_estimators=20, random_state=42)  # fewer trees
            xgb = XGBRegressor(n_estimators=20, random_state=42, verbosity=0)
            
            lr.fit(X_train, y_train)
            rf.fit(X_train, y_train)
            xgb.fit(X_train, y_train)
            
            # Predict on test
            lr_pred  = lr.predict(X_test)
            rf_pred  = rf.predict(X_test)
            xgb_pred = xgb.predict(X_test)
            
            ensemble_test_pred = (lr_pred + rf_pred + xgb_pred)/3
            
            # Store in df
            df_feat.loc[test_index, f"Pred_{hname}"] = ensemble_test_pred
            
            # Plot
            plot_test_predictions(hname, y_test, ensemble_test_pred, test_index)
            
            # Future
            lr_fut  = lr.predict(X_last_scaled)
            rf_fut  = rf.predict(X_last_scaled)
            xgb_fut = xgb.predict(X_last_scaled)
            fut_ensemble = (lr_fut + rf_fut + xgb_fut)/3
            future_preds[hname] = fut_ensemble[0]
        
        # Show final numeric
        st.write("## Future Predictions (After Last Known Date)")
        for hname in horizon_map:
            st.write(f"- **{hname}**: {future_preds[hname]:.2f} USD")
        
    else:
        st.info("Enter a ticker, then click Predict.")

def feature_engineering(df):
    df = df.copy()
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up/ema_down
    df['RSI'] = 100 - (100/(1+rs))
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    df.dropna(inplace=True)
    return df

def plot_test_predictions(horizon_name, y_test, pred, test_index):
    st.write(f"**Backtest: {horizon_name}** - Actual vs. Ensemble Prediction")
    fig, ax = plt.subplots(figsize=(8,3))
    actual = y_test.values
    ax.plot(test_index, actual, label='Actual', color='blue')
    ax.plot(test_index, pred, label='Predicted', color='red')
    ax.set_title(f"{horizon_name}: Actual vs. Predicted")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
