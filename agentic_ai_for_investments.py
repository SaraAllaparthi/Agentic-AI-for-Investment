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

---

## **3. Step-by-Step Instructions to Run the App**

### **a. Set Up the Environment**

1. **Ensure Python is Installed**: Make sure you have Python 3.7 or higher installed on your system. You can check your Python version by running:

    ```bash
    python --version
    ```

2. **Create a Virtual Environment (Optional but Recommended)**:

    ```bash
    python -m venv stock_env
    ```

    Activate the virtual environment:

    - **Windows**:

        ```bash
        stock_env\Scripts\activate
        ```

    - **macOS/Linux**:

        ```bash
        source stock_env/bin/activate
        ```

### **b. Install Dependencies**

1. **Create `requirements.txt`**:

    Create a file named `requirements.txt` in your project directory with the following content:

    ```txt
    streamlit
    yfinance
    pandas
    matplotlib
    prophet
    ```

2. **Install Dependencies**:

    Run the following command in your terminal or command prompt to install the necessary libraries:

    ```bash
    pip install -r requirements.txt
    ```

    **Note**: Installing `prophet` can sometimes be challenging due to its dependencies. If you encounter issues, consider using `conda`:

    ```bash
    conda install -c conda-forge prophet
    ```

    Or refer to [Prophet's official installation guide](https://facebook.github.io/prophet/docs/installation.html) for detailed instructions based on your operating system.

### **c. Save the Corrected `app.py` Code**

1. **Create `app.py`**:

    In your project directory, create a file named `app.py` and **paste the corrected code** provided above **exactly as is**, ensuring that no additional text or formatting is included.

2. **Verify the Code**:

    Open `app.py` in your preferred code editor (e.g., VS Code, Sublime Text) and ensure that:

    - All comments start with `#`.
    - There are no unintended bullet points or Markdown syntax within the code.
    - All quotation marks are properly closed.

### **d. Run the Streamlit App**

1. **Navigate to the Project Directory**:

    In your terminal or command prompt, navigate to the directory containing `app.py`:

    ```bash
    cd path_to_your_project_directory
    ```

2. **Run the App**:

    Execute the following command to run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

    This command should open a new tab in your default web browser displaying the Streamlit app. If it doesn't open automatically, check the terminal for a local URL (usually `http://localhost:8501`) and navigate to it manually.

### **e. Interact with the App**

1. **View Available Tickers**:

    On the left sidebar, you'll see a list of available tickers with their corresponding company names for reference.

2. **Enter a Stock Ticker**:

    In the main interface, locate the input field labeled "Enter Stock Ticker Symbol (e.g., AAPL, GOOGL):". Enter a valid ticker symbol from the sidebar list (e.g., `AAPL` for Apple Inc.).

3. **Initiate Forecasting**:

    Click the **"Forecast 3-Month Price"** button. The app will:

    - Fetch the last 3 years of historical data for the entered ticker.
    - Perform forecasting using the Prophet model.
    - Display the predicted average closing price for the next 3 months.
    - Show two plots:
        - **Stock Price Forecast**: Actual vs. Forecasted Prices with confidence intervals.
        - **Last 6 Months: Actual vs. Forecasted Prices**: A focused comparison for the recent 6 months.

4. **Review Results**:

    - **Predicted Price**: Located below the first plot, showing the average predicted closing price in 3 months.
    - **Plots**: Visual representations of historical and forecasted data for better insight.

---

## **4. Additional Recommendations and Best Practices**

### **a. Ensure Proper Code Copying**

When copying the `app.py` code:

- **Only Copy Within the Code Block**: Ensure that you copy **only** the code within the triple backticks (```) and **exclude** any explanatory text or bullet points outside the code block.
  
- **Avoid Extra Characters**: Do not include any additional characters or lines outside the code block to prevent syntax errors.

### **b. Debugging Tips**

If you still encounter issues:

1. **Check Streamlit Logs**:

    - Streamlit provides detailed error logs in the terminal where you ran `streamlit run app.py`. Review these logs for specific error messages.

2. **Use `st.write()` for Debugging**:

    - Insert `st.write()` statements at various points in your code to display variable values and ensure data is being processed correctly.

    ```python
    st.write(df_prophet.head())
    ```

3. **Validate Data**:

    - Ensure that the fetched data contains the necessary columns (`Date` and `Close`) and that the data isn't empty after processing.

### **c. Enhance User Experience**

1. **Autocomplete for Tickers**:

    - Implement an autocomplete feature to help users select valid ticker symbols. This can reduce input errors.

    ```python
    ticker_input = st.selectbox(
        "Enter Stock Ticker Symbol:",
        options=list(AVAILABLE_TICKERS.keys()),
        format_func=lambda x: f"{x} - {AVAILABLE_TICKERS[x]}"
    )
    ```

2. **Download Forecast Data**:

    - Allow users to download the forecasted data as a CSV file for further analysis.

    ```python
    st.download_button(
        label="Download Forecast Data",
        data=forecast.to_csv(index=False),
        file_name=f"{ticker_input}_forecast.csv",
        mime='text/csv',
    )
    ```

### **d. Optimize Performance**

1. **Caching Prophet Models**:

    - Prophet models can be computationally intensive. Consider caching the fitted model if users frequently forecast the same ticker.

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

    **Note**: `@st.cache_resource` is available in Streamlit version 1.18 and above.

2. **Adjust Forecast Horizon**:

    - If performance is an issue, consider reducing the forecast period (e.g., 60 days instead of 90).

    ```python
    future = model.make_future_dataframe(periods=60)
    predicted_price = forecast.iloc[-60:]['yhat'].mean()
    ```

### **e. Regular Updates and Maintenance**

- **Keep Dependencies Updated**: Regularly update your Python libraries to benefit from the latest features and bug fixes.

    ```bash
    pip install --upgrade streamlit yfinance pandas matplotlib prophet
    ```

- **Monitor App Performance**: Use Streamlit's built-in performance monitoring or add custom logging to identify and address bottlenecks.

---

## **5. Final Notes**

- **Model Limitations**: Stock price forecasting is inherently uncertain due to market volatility and numerous external factors. While Prophet can capture trends and seasonality, it cannot account for sudden market shifts or unforeseen events. Use the predictions as one of many tools in your decision-making process.

- **Continuous Improvement**: Regularly update the app based on user feedback and evolving requirements to enhance functionality and user experience.

- **User Feedback**: Encourage users to provide feedback to continually refine and improve the app's functionality and user experience.

---

By following the corrected code and these comprehensive instructions, your Streamlit app should function correctly, allowing you to input any valid ticker symbol, perform forecasting, and visualize the results without encountering syntax or runtime errors. If you continue to experience issues, please provide the updated error messages or specific sections of your code where the errors occur, and I'll be happy to assist further!
