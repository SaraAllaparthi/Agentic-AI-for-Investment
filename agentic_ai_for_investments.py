# streamlit_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from textblob import TextBlob
import feedparser

# --- Helper functions ---

def fetch_stock_data(ticker, period="5y"):
    """
    Fetch historical stock data from yfinance.
    """
    data = yf.download(ticker, period=period)
    data.dropna(inplace=True)
    return data

def add_technical_indicators(data):
    """
    Add a few simple technical indicators: 10-day MA, RSI, and MACD.
    Assumes 'Close' column exists.
    """
    # 10-day Moving Average
    data['MA10'] = data['Close'].rolling(window=10).mean()

    # RSI Calculation
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD Calculation
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26

    data.dropna(inplace=True)
    return data

def fetch_news_sentiment(ticker):
    """
    Fetch recent news headlines from a public RSS feed (e.g., Google News)
    and compute an average sentiment score using TextBlob.
    """
    # Use Google News RSS feed for the stock (this is a simple example)
    # You can modify the query as needed.
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock"
    feed = feedparser.parse(rss_url)
    sentiments = []
    for entry in feed.entries:
        headline = entry.title
        analysis = TextBlob(headline)
        sentiments.append(analysis.sentiment.polarity)
    if sentiments:
        avg_sentiment = np.mean(sentiments)
    else:
        avg_sentiment = 0.0
    return avg_sentiment

def train_time_index_model(data):
    """
    Train a Linear Regression model on a simple time index versus the closing price.
    This will be used to forecast future prices.
    """
    df = data.copy().reset_index()
    # Create a numeric time index (number of days since the start)
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Days']]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    # Evaluate performance (for reference)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return model, mse, r2, df

def forecast_future_prices(model, last_day, forecast_days):
    """
    Given a trained model and the last day as an integer, forecast the next forecast_days.
    """
    future_days = np.array([last_day + i for i in range(1, forecast_days+1)]).reshape(-1, 1)
    future_pred = model.predict(future_days)
    return future_days, future_pred

# --- Streamlit App ---

st.title("Agentic AI Stock Price Predictor")
st.write("""
### Predict the next 3 months of closing prices for a stock.
Enter a stock ticker (e.g., AAPL, MSFT, GOOGL) to see today's price and a 3‑month forecast.
""")

# Sidebar for user inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")

# Fetch and display data only if a ticker is provided
if ticker:
    st.write(f"Fetching data for **{ticker.upper()}** ...")
    data = fetch_stock_data(ticker)
    if data.empty:
        st.error("No data found. Please check the ticker symbol.")
    else:
        data = add_technical_indicators(data)

        # Display the latest (today's) closing price
        latest_date = data.index[-1]
        latest_price = data['Close'].iloc[-1]
        st.write(f"### Today's closing price ({latest_date.date()}): ${latest_price:.2f}")

        # Show a brief chart of historical closing prices
        st.write("#### Historical Closing Prices")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data.index, data['Close'], label="Closing Price", color='blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Get sentiment score from recent news headlines
        st.write("#### Recent News Sentiment")
        sentiment = fetch_news_sentiment(ticker)
        st.write(f"Average sentiment score (−1 very negative, 1 very positive): **{sentiment:.2f}**")

        # --- Forecasting ---

        st.write("#### Forecasting the Next 3 Months")

        # For simplicity, we forecast using a Linear Regression on a time index.
        # (In future iterations you can integrate technical indicators and sentiment into the model.)
        model, mse, r2, df_model = train_time_index_model(data)
        st.write(f"Model Performance on historical data: **MSE = {mse:.2f}**, **R² = {r2:.2f}**")

        # Forecast horizon: approximately 3 months ~ 60 trading days.
        forecast_days = 60
        last_day = df_model['Days'].iloc[-1]
        future_days, future_prices = forecast_future_prices(model, last_day, forecast_days)

        # Create future date range (skipping weekends can be added later; for now, use calendar days)
        last_date = df_model['Date'].iloc[-1]
        future_dates = [last_date + datetime.timedelta(days=int(i)) for i in range(1, forecast_days+1)]
        forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_prices})

        # Plot the forecast along with historical data
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(data.index, data['Close'], label="Historical Close", color="blue")
        ax2.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast (Next 3 Months)", color="red", linestyle="--")
        ax2.set_title(f"{ticker.upper()} Stock Price Forecast")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price (USD)")
        ax2.legend()
        ax2.grid(True)
        # Improve date formatting on x-axis
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        st.write("### Disclaimer")
        st.write("This prediction is for informational purposes only and should not be considered financial advice.")
