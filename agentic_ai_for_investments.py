Agentic AI to Stock Price Prediction
Enter a ticker symbol (as per Yahoo Finance) to see predictions for the next 1 day, 1 week, 1 month, and 6 months.

TypeError: Importing a module script failed.
Fetching data for GOOGL...

Historical Stock Price
TypeError: Importing a module script failed.
Loaded pre-trained model.

Predicted Prices
Next 1 Day: $194.98

Next 1 Week: $195.41

Next 1 Month: $190.61

Next 6 Months: $170.36

ValueError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/agentic-ai-for-investment/agentic_ai_for_investments.py", line 159, in <module>
    insight = get_market_insight(data)
              ^^^^^^^^^^^^^^^^^^^^^^^^
File "/mount/src/agentic-ai-for-investment/agentic_ai_for_investments.py", line 105, in get_market_insight
    if pd.isna(latest_ma50):
       ^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/pandas/core/generic.py", line 1577, in __nonzero__
    raise ValueError(
