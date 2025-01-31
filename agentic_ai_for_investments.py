import streamlit as st
import yfinance as yf
import pandas as pd
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

# Function to fetch data with error handling
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

### **Key Corrections and Enhancements**

1. **Proper Commenting and Removal of Markdown Syntax:**
    - All comments within the code are prefixed with `#`.
    - Removed any bullet points or Markdown syntax from the code to prevent Python from misinterpreting them as code.
    - Ensured that explanatory texts are either within comment lines or in Markdown outside of code blocks.

2. **Enhanced Error Handling:**
    - **Data Fetching Errors:** Wrapped the data fetching in a `try-except` block to catch and display errors if `yfinance` fails.
    - **Data Integrity Checks:** After renaming columns, checked for the existence of `'Date'` and `'Close'` columns. Converted `'y'` to numeric and handled non-numeric entries by dropping them and notifying the user.
    - **Model Fitting and Prediction Errors:** Wrapped Prophet model fitting and prediction steps in `try-except` blocks to catch and display any errors.
    - **Forecast Data Availability:** Added checks to ensure there's sufficient forecasted data to plot the last 6 months' comparison.

3. **User Interface (UI) Improvements:**
    - **Available Tickers Sidebar:** Clearly lists available tickers with their corresponding company names to guide users.
    - **Ticker Input Field:** Provides a text input for users to enter any ticker symbol, validated against the predefined `AVAILABLE_TICKERS` list.
    - **Forecast Button:** Users must click the **"Forecast 3-Month Price"** button to initiate the forecasting process, preventing automatic runs on every input change.

4. **Visualization Enhancements:**
    - **Confidence Intervals:** Added shaded areas representing Prophet's confidence intervals for better visualization of uncertainty.
    - **Date Formatting:** Improved date labels on the x-axis for clarity and readability.
    - **Separate Plots:** Created two distinct plots:
        - **Overall Forecast:** Shows historical data, forecasted prices, and confidence intervals.
        - **Last 6 Months Comparison:** Provides a focused view comparing the last 6 months' actual prices with forecasted prices.

5. **Streamlit Best Practices:**
    - **Explicit Figure Passing:** Passed the `figure` object directly to `st.pyplot(fig)` to adhere to Streamlit's latest best practices, avoiding the need for deprecated options.
    - **Caching:** Utilized `@st.cache_data` to cache fetched data, reducing load times for repeated ticker forecasts.

### **3. Running the Corrected Streamlit App**

1. **Ensure Dependencies are Installed:**
    
    Make sure all required libraries are installed. If you created the `requirements.txt`, run:
    
    ```bash
    pip install -r requirements.txt
    ```
    
    *Note: If you encounter issues installing `prophet`, consider using `conda` or refer to [Prophet's official installation guide](https://facebook.github.io/prophet/docs/installation.html).*

2. **Save the Corrected Code:**
    
    Save the provided `app.py` code in your project directory.

3. **Run the Streamlit App:**
    
    In your terminal, navigate to the project directory and execute:
    
    ```bash
    streamlit run app.py
    ```
    
4. **Interact with the App:**
    
    - **Enter a Stock Ticker:** In the main interface, input a stock ticker symbol (e.g., `AAPL`, `GOOGL`).
    - **Click the Forecast Button:** Press the **"Forecast 3-Month Price"** button to initiate data fetching and forecasting.
    - **View Results:** Observe the forecasted price and the comparison graphs.

### **4. Additional Recommendations**

1. **Validate Data Before Processing:**
    
    While the app now includes robust error handling, it's good practice to validate the data's integrity before proceeding with forecasting. Consider adding more checks if necessary.

2. **Improve User Experience:**
    
    - **Autocomplete for Tickers:** Implement an autocomplete feature for ticker input to guide users.
    - **Download Forecast Data:** Allow users to download the forecasted data as a CSV file.
    
    ```python
    st.download_button(
        label="Download Forecast Data",
        data=forecast.to_csv(index=False),
        file_name=f"{ticker_input}_forecast.csv",
        mime='text/csv',
    )
    ```

3. **Optimize Performance:**
    
    - **Caching Prophet Models:** Prophet models can be computationally intensive. If you plan to allow multiple forecasts for the same ticker, consider caching the fitted model to avoid retraining.
    
    ```python
    @st.cache_resource
    def get_prophet_model(df):
        model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
        model.fit(df)
        return model
    ```
    
    Then use:
    
    ```python
    model = get_prophet_model(df_prophet)
    ```

4. **Interactive Plots:**
    
    Utilize interactive plotting libraries like Plotly for enhanced user interaction.

5. **Regular Updates:**
    
    Periodically update the app and retrain models with the latest data to maintain forecast accuracy.

### **5. Final Notes**

- **Model Limitations:** Stock price forecasting is inherently uncertain due to market volatility and numerous external factors. While Prophet can capture trends and seasonality, it cannot account for sudden market shifts or unforeseen events.

- **Continuous Improvement:** Regularly update the app based on user feedback and evolving requirements to enhance functionality and user experience.

- **User Feedback:** Encourage users to provide feedback to continually refine and improve the app's functionality and user experience.

By following the corrected code and these guidelines, your Streamlit app should function correctly, allowing you to input any valid ticker symbol, perform forecasting, and visualize the results without encountering syntax or runtime errors. If you continue to experience issues, please provide the updated error messages or specific sections of your code where the errors occur, and I'll be happy to assist further!
