# simple_agent_app.py

import streamlit as st
from agno.agent import Assistant
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools

# Set up the Streamlit app
st.title("AI Investment Agent 📈🤖")
st.caption("Get the latest stock price and company info with a single query.")

# Get OpenAI API key from user (make sure you have one)
openai_api_key = st.text_input("OpenAI API Key", type="password")

# Input field for the stock symbol (default AAPL)
stock_symbol = st.text_input("Enter the stock symbol", value="AAPL")

# Proceed if both the API key and stock symbol are provided
if openai_api_key and stock_symbol:
    # Create an instance of the Assistant using the OpenAIChat LLM
    assistant = Assistant(
        llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
        tools=[YFinanceTools(stock_price=True, company_info=True)],
        show_tool_calls=True,
    )
    
    # Build a query for the assistant
    query = f"Provide the current stock price and basic company info for {stock_symbol}."
    
    # Run the assistant and get the response
    response = assistant.run(query, stream=False)
    
    # Display the response content on the Streamlit app
    st.write(response.content)
