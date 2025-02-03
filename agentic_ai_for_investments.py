import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
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

@st.cache_data(ttl=60)
def fetch_data(ticker):
    """Fetch historical data for the given ticker."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Flatten columns if they are a MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[-1] for col in data.columns.values]
    
    return data

data_load_state = st.text("Fetching data...")
data = fetch_data(ticker)
data_load_state.text("")

if data.empty:
    st.warning("No data returned for this ticker. Data might be missing from Yahoo Finance.")
else:
    st.subheader(f"Historical Data for {ticker}")
    st.write(data.tail())

    # Prepare data for Prophet by resetting index and selecting only the needed columns.
    df = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    
    # Ensure the 'ds' column is datetime and 'y' is numeric.
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    # Convert the 'y' column to numeric. Now that the columns are flattened, df['y'] should be a 1D Series.
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    # Drop rows with missing values in either 'ds' or 'y'.
    df.dropna(subset=['ds', 'y'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    st.write("Data used for training (first few rows):")
    st.write(df.head())
    
    # Train the Prophet model
    st.write("Training the forecasting model...")
    try:
        model = Prophet()
        model.fit(df)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()
    
    # Create a DataFrame for future predictions (approx. 6 months = 180 days)
    future = model.make_future_dataframe(periods=180)
    forecast = model.predict(future)
    
    # Plot the forecast using Prophet's built-in Plotly integration.
    st.subheader("Forecasted Stock Prices (Next 6 Months)")
    forecast_fig = plot_plotly(model, forecast)
    forecast_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend_title="Legend"
    )
    st.plotly_chart(forecast_fig, use_container_width=True)
    
    # Create an interactive comparison chart between historical and predicted prices.
    st.subheader("Comparison: Actual vs Predicted Prices")
    last_actual_date = df["ds"].max()
    forecast_future = forecast[forecast["ds"] > last_actual_date]
    
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
    
    st.markdown(
        """
        **Model Explanation:**
        - The model used is [Prophet](https://facebook.github.io/prophet/), a forecasting tool that uses an additive model.
        - The forecast includes a prediction for the next 6 months of stock prices, with the blue line representing historical actual prices and the red line representing the forecasted prices.
        - This model uses only raw closing price data and is retrained weekly.
        """
    )
