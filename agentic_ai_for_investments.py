import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

from prophet import Prophet

import matplotlib.dates as mdates

# Remove or comment out the problematic line
# st.set_option('deprecation.showPyplotGlobalUse', False)

# Caching the ticker list to speed up the app
@st.cache_data
def load_ticker_list():
    # Curated list of tickers
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

def main():
    st.title("📈 Stock Price Forecasting - Next 3 Months")

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

    # 3. Prepare Data for Prophet
    df_prophet = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # 4. Ensure 'y' is numeric
    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
    df_prophet.dropna(inplace=True)  # Drop rows with non-numeric 'y'

    # 5. Initialize and Fit Prophet Model
    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_prophet)

    # 6. Create Future DataFrame for 3 Months (approx. 90 days)
    future = model.make_future_dataframe(periods=90)

    # 7. Make Predictions
    forecast = model.predict(future)

    # 8. Extract the Predicted Price 3 Months Ahead
    predicted_price = forecast.iloc[-90:]['yhat'].mean()

    # 9. Plotting
    st.write("## Stock Price Forecast")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot historical data
    ax.plot(df_prophet['ds'], df_prophet['y'], label='Actual Price', color='blue')

    # Plot forecast
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Price', color='red')

    # Highlight the forecasted area
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                    color='pink', alpha=0.3, label='Confidence Interval')

    # Formatting the plot
    ax.set_title(f"{selected_ticker} - Actual vs. Forecasted Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()

    # Improve date formatting on the x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Pass the figure to st.pyplot
    st.pyplot(fig)

    # 10. Display Predicted Price
    st.write(f"### 📅 **Predicted Average Closing Price in 3 Months**: **${predicted_price:.2f} USD**")

    # 11. Plot Last 6 Months Actual vs Forecasted
    st.write("## Last 6 Months: Actual vs. Forecasted Prices")

    # Define the period for the last 6 months
    six_months_ago = end_date - timedelta(days=6*30)  # Approx. 6 months
    six_months_ago_pd = pd.Timestamp(six_months_ago)

    mask = (forecast['ds'] >= six_months_ago_pd) & (forecast['ds'] <= end_date + timedelta(days=90))
    plot_forecast = forecast.loc[mask]

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.plot(df_prophet['ds'], df_prophet['y'], label='Historical Price', color='blue')
    ax2.plot(plot_forecast['ds'], plot_forecast['yhat'], label='Forecasted Price', color='red')
    ax2.fill_between(plot_forecast['ds'], plot_forecast['yhat_lower'], plot_forecast['yhat_upper'], 
                    color='pink', alpha=0.3, label='Confidence Interval')
    ax2.set_title(f"{selected_ticker} - Last 6 Months: Actual vs. Forecasted Prices")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Pass the figure to st.pyplot
    st.pyplot(fig2)

if __name__ == "__main__":
    main()
