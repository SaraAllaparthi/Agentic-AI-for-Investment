import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta

# Set the page configuration
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("Stock Price Predictor Dashboard")
st.write(
    """
    This dashboard fetches the last 5 years of historical stock price data from Yahoo Finance,
    uses a time series forecasting model (Prophet) to predict stock prices for the next 6 months,
    and displays a comparison of actual vs. predicted stock prices.
    """
)

# Sidebar for user inputs
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, GOOGL)", value="AAPL")

@st.cache_data(ttl=60)  # cache data for 60 seconds to prevent hitting rate limits
def fetch_data(ticker):
    # Define the date range: last 5 years
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Fetch the historical data
data_load_state = st.text("Fetching data...")
data = fetch_data(ticker)
data_load_state.text("")

if data.empty:
    st.warning("No data returned for this ticker. Data might be missing from Yahoo Finance.")
else:
    st.subheader(f"Historical Data for {ticker}")
    st.write(data.tail())

    # Prepare the data for Prophet
    # Prophet requires a dataframe with columns "ds" (datetime) and "y" (value)
    df = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    
    # Create and train the Prophet model
    st.write("Training the forecasting model...")
    model = Prophet()
    try:
        model.fit(df)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

    # Create a dataframe to hold predictions for the next 6 months
    future = model.make_future_dataframe(periods=6*30)  # approx 6 months (6*30 days)
    forecast = model.predict(future)

    # Plot forecast using Prophet's built-in plotly integration
    st.subheader("Forecasted Stock Prices (Next 6 Months)")
    forecast_fig = plot_plotly(model, forecast)
    forecast_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend_title="Legend"
    )
    st.plotly_chart(forecast_fig, use_container_width=True)

    # Create an interactive comparison chart between historical and predicted prices
    st.subheader("Comparison: Actual vs Predicted Prices")

    # Prepare data: actual historical data and predicted forecast
    # We'll only show predictions that are in the future (beyond the last actual date)
    last_actual_date = df["ds"].max()
    forecast_future = forecast[forecast["ds"] > last_actual_date]

    # Create traces for actual and forecasted data
    trace_actual = go.Scatter(
        x=df["ds"],
        y=df["y"],
        mode="lines",
        name="Actual Price",
        line=dict(color="blue")
    )
    trace_predicted = go.Scatter(
        x=forecast_future["ds"],
        y=forecast_future["yhat"],
        mode="lines",
        name="Predicted Price",
        line=dict(color="red")
    )

    layout = go.Layout(
        title=f"Actual vs Predicted Prices for {ticker}",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        hovermode="x unified"
    )

    fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    # Display additional explanation
    st.markdown(
        """
        **Model Explanation:**
        - The model used is [Prophet](https://facebook.github.io/prophet/), which is a procedure for forecasting time series data based on an additive model.
        - The forecast includes a prediction for the next 6 months of stock prices, where the blue line represents historical actual prices and the red line represents the forecasted prices.
        - Note that this model uses only the raw closing price data and is retrained weekly.
        """
    )
