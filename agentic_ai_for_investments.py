import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from prophet import Prophet
import matplotlib.dates as mdates

# Set Streamlit page configuration
st.set_page_config(
    page_title="Stock Price Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define a dictionary of available tickers for reference
AVAILABLE_TICKERS = {
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

# Sidebar: Display available tickers
st.sidebar.header("Available Tickers")
st.sidebar.write("Enter a ticker symbol from the list below:")
for ticker, name in AVAILABLE_TICKERS.items():
    st.sidebar.write(f"**{ticker}**: {name}")

# Function to fetch data
@st.cache_data(show_spinner=False)
def fetch_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def main():
    st.title("📈 Stock Price Forecasting - Next 3 Months")
    
    st.markdown("""
    **Disclaimer**: This application is for **informational purposes only** and does **not** constitute financial advice. 
    Always do your own research or consult a professional before making investment decisions.
    """)

    # Ticker input
    ticker_input = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, GOOGL):", value="AAPL").upper().strip()
    
    # Button to initiate forecasting
    if st.button("Forecast 3-Month Price"):
        if ticker_input not in AVAILABLE_TICKERS:
            st.error("Invalid ticker symbol. Please enter a valid ticker from the sidebar list.")
            return

        with st.spinner("Fetching data and performing forecasting..."):
            end_date = date.today()
            start_date = end_date - timedelta(days=3*365)  # Last 3 years
            data = fetch_data(ticker_input, start_date, end_date)
        
        if data.empty:
            st.error("No data found for the entered ticker. Please try a different ticker.")
            return

        # Prepare data for Prophet
        try:
            df_prophet = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        except KeyError:
            st.error("The fetched data does not contain 'Date' or 'Close' columns. Please try a different ticker.")
            return

        # Ensure 'y' is numeric and handle any non-numeric entries
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        initial_length = len(df_prophet)
        df_prophet.dropna(inplace=True)  # Drop rows with non-numeric 'y'
        final_length = len(df_prophet)
        
        if df_prophet.empty:
            st.error("Insufficient numeric data after processing. Please try a different ticker.")
            return
        elif final_length < initial_length:
            st.warning(f"Dropped {initial_length - final_length} rows due to non-numeric 'Close' values.")

        # Initialize and fit Prophet model
        try:
            model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
            model.fit(df_prophet)
        except Exception as e:
            st.error(f"An error occurred while fitting the Prophet model: {e}")
            return

        # Create future dataframe for 3 months (approx. 90 days)
        future = model.make_future_dataframe(periods=90)

        # Make predictions
        try:
            forecast = model.predict(future)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            return

        # Extract the predicted price 3 months ahead
        predicted_price = forecast.iloc[-90:]['yhat'].mean()

        # Plotting: Actual vs Forecasted Prices
        st.write("## Stock Price Forecast")
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot historical data
        ax.plot(df_prophet['ds'], df_prophet['y'], label='Actual Price', color='blue')

        # Plot forecasted data
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Price', color='red')

        # Fill confidence interval
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                        color='pink', alpha=0.3, label='Confidence Interval')

        # Formatting the plot
        ax.set_title(f"{AVAILABLE_TICKERS[ticker_input]} ({ticker_input}) - Actual vs. Forecasted Prices")
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

        # Display Predicted Price
        st.write(f"### 📅 **Predicted Average Closing Price in 3 Months**: **${predicted_price:.2f} USD**")

        # Plot Last 6 Months: Actual vs Forecasted
        st.write("## Last 6 Months: Actual vs. Forecasted Prices")
        six_months_ago = end_date - timedelta(days=6*30)  # Approx. 6 months
        six_months_ago_pd = pd.Timestamp(six_months_ago)

        mask = (forecast['ds'] >= six_months_ago_pd) & (forecast['ds'] <= end_date + timedelta(days=90))
        plot_forecast = forecast.loc[mask]

        if plot_forecast.empty:
            st.warning("Not enough forecasted data to display the last 6 months comparison.")
            return

        fig2, ax2 = plt.subplots(figsize=(14, 7))
        ax2.plot(df_prophet['ds'], df_prophet['y'], label='Historical Price', color='blue')
        ax2.plot(plot_forecast['ds'], plot_forecast['yhat'], label='Forecasted Price', color='red')
        ax2.fill_between(plot_forecast['ds'], plot_forecast['yhat_lower'], plot_forecast['yhat_upper'], 
                        color='pink', alpha=0.3, label='Confidence Interval')

        # Formatting the plot
        ax2.set_title(f"{AVAILABLE_TICKERS[ticker_input]} ({ticker_input}) - Last 6 Months: Actual vs. Forecasted Prices")
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
    ```

#### **c. Explanation of the Corrections and Enhancements**

1. **Improved Error Handling**:
    - **Fetching Data**:
        ```python
        try:
            df_prophet = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        except KeyError:
            st.error("The fetched data does not contain 'Date' or 'Close' columns. Please try a different ticker.")
            return
        ```
        Ensures that the necessary columns `'Date'` and `'Close'` exist in the fetched data.

    - **Converting 'y' to Numeric**:
        ```python
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet.dropna(inplace=True)  # Drop rows with non-numeric 'y'
        
        if df_prophet.empty:
            st.error("Insufficient numeric data after processing. Please try a different ticker.")
            return
        elif final_length < initial_length:
            st.warning(f"Dropped {initial_length - final_length} rows due to non-numeric 'Close' values.")
        ```
        Converts the `'y'` column to numeric and handles non-numeric entries gracefully by dropping them and notifying the user.

    - **Prophet Model Fitting and Prediction**:
        Wrapped in `try-except` blocks to catch and display any errors during model fitting and prediction.

    - **Forecast Data Availability**:
        ```python
        if plot_forecast.empty:
            st.warning("Not enough forecasted data to display the last 6 months comparison.")
            return
        ```
        Ensures that there's enough data to plot the last 6 months comparison.

2. **User Interface Enhancements**:
    - **Page Configuration**:
        ```python
        st.set_page_config(
            page_title="Stock Price Forecasting",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        ```
        Sets the page title, layout, and initial sidebar state for a better user experience.

    - **Available Tickers Sidebar**:
        Clearly lists available tickers with their corresponding company names to guide users.

    - **Input Field and Button**:
        Provides a text input for the ticker symbol and a button to initiate forecasting.

3. **Ensuring 'y' is a Series**:
    - By selecting a single column `['y']` from the DataFrame, we ensure that `df_prophet['y']` is a Pandas Series, which is compatible with `pd.to_numeric`.
    - Additional error checks confirm that after processing, `df_prophet` is not empty.

4. **Visualization Enhancements**:
    - **Confidence Intervals**: Added shaded areas representing Prophet's confidence intervals.
    - **Date Formatting**: Improved date labels for clarity.
    - **Figure Passing**: Explicitly passes the `figure` object to `st.pyplot(fig)` to adhere to Streamlit's latest best practices.

5. **Streamlined Flow**:
    - The app only proceeds with forecasting when the user clicks the **"Forecast 3-Month Price"** button.
    - Early returns prevent the app from attempting to process invalid inputs or insufficient data.

### **3. Running the Corrected Streamlit App**

1. **Ensure Dependencies are Installed**

   Make sure all required libraries are installed. If you created the `requirements.txt`, run:

   ```bash
   pip install -r requirements.txt
