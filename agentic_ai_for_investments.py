import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

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

    # 7. Model Training
    st.write("## Training Models...")

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
    }

    predictions = {}
    mse_scores = {}
    r2_scores = {}

    for model_name, model in models.items():
        with st.spinner(f"Training {model_name}..."):
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            predictions[model_name] = preds
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            mse_scores[model_name] = mse
            r2_scores[model_name] = r2

    # 8. Model Evaluation
    st.write("## Model Evaluation on Test Set")

    eval_df = pd.DataFrame({
        "Model": list(models.keys()),
        "MSE": [mse_scores[model] for model in models.keys()],
        "R² Score": [r2_scores[model] for model in models.keys()]
    })

    st.table(eval_df.style.format({"MSE": "{:.2f}", "R² Score": "{:.4f}"}))

    # 9. Select Best Model
    best_model_name = min(mse_scores, key=mse_scores.get)
    best_model = models[best_model_name]
    best_pred = predictions[best_model_name]

    st.write(f"### 🏆 Best Model: **{best_model_name}** with **MSE = {mse_scores[best_model_name]:.2f}** and **R² = {r2_scores[best_model_name]:.4f}**")

    # 10. Plot Actual vs. Predicted on Test Set
    st.write("## Test Set: Actual vs. Predicted")

    plot_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': best_pred
    }, index=y_test.index)

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

    # 11. Predict Future Prices (~3 Months Ahead)
    st.write("## 📈 Future Price Prediction (~3 Months Ahead)")

    last_row = data.iloc[-1][feature_cols].values.reshape(1, -1)
    last_row_scaled = scaler.transform(last_row)

    future_pred = best_model.predict(last_row_scaled)[0]

    # Optionally, you can extend predictions iteratively for multiple days
    # For simplicity, we're predicting a single point ~3 months ahead

    st.write(f"**Predicted Closing Price ~3 Months After {selected_ticker} on {y_test.index[-1].date() + timedelta(days=63)}**: **${future_pred:.2f}**")

    # 12. Plot Future Prediction on Top of Historical Data
    st.write("## 📊 Historical Prices with Future Prediction")

    # Get the last 6 months of data for reference
    six_months_ago = end_date - timedelta(days=6*30)  # Approx. 6 months
    historical_df = data[data.index >= six_months_ago]

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.plot(historical_df.index, historical_df['Close'], label='Historical Close', color='blue')
    ax2.scatter(y_test.index[-1] + timedelta(days=63), future_pred, label='3-Month Prediction', color='green', marker='X', s=100)
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
