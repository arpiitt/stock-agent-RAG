import streamlit as st
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time

# Simple chat interface for stock information
st.set_page_config(page_title="Stock Agent", layout="wide")

# Function to get real-time stock data directly from Yahoo Finance
def get_realtime_stock_data(ticker, period="1mo"):
    try:
        # Get data directly from Yahoo Finance
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        # Reset index to make date a column
        hist = hist.reset_index()
        
        # Rename columns to match expected format
        hist = hist.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Add ticker column for consistency with local data
        if 'ticker' not in hist.columns:
            hist['ticker'] = ticker
            
        return hist
    except Exception as e:
        st.error(f"Error fetching real-time data: {e}")
        # Fall back to local data
        return get_local_stock_history(ticker)

# Function to get historical stock data from local files
def get_local_stock_history(ticker, days=30):
    try:
        # Path to CSV file
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        csv_path = os.path.join(data_dir, f"{ticker}_history.csv")
        
        # Check if file exists
        if os.path.exists(csv_path):
            # Load data from CSV
            df = pd.read_csv(csv_path)
            
            # Convert date column
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Filter for recent days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Filter and sort - convert to datetime for comparison
            df = df[df['Date'] >= start_date]
            df = df.sort_values('Date', ascending=False)
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            return df
        else:
            # Try the combined file
            combined_path = os.path.join(data_dir, "all_stocks_history.csv")
            if os.path.exists(combined_path):
                df = pd.read_csv(combined_path)
                df = df[df['ticker'] == ticker]
                
                # Convert date column
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Filter for recent days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Filter and sort - convert to datetime for comparison
                df = df[df['Date'] >= start_date]
                df = df.sort_values('Date', ascending=False)
                
                # Rename columns to match expected format
                df = df.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                return df
            else:
                st.warning(f"No local data found for {ticker}.")
                return None
    except Exception as e:
        st.error(f"Error loading stock data: {e}")
        return None

# Function to get XAUUSD (Gold) price data
def get_gold_price_data(period="1mo"):
    try:
        # Yahoo Finance ticker for Gold
        gold = yf.Ticker("GC=F")
        history = gold.history(period=period)
        history = history.reset_index()
        history = history.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        history['ticker'] = "GOLD"
        return history
    except Exception as e:
        st.error(f"Error fetching Gold price data: {e}")
        return None

# Function to get historical stock data (tries real-time first, then falls back to local)
def get_stock_history(ticker, days=30, use_realtime=True):
    # Handle Gold price data specially
    if ticker == "GOLD" or ticker == "GC=F" or ticker == "XAUUSD":
        return get_gold_price_data(period=f"{days}d")
        
    # Try to get real-time data first if enabled
    if use_realtime:
        df = get_realtime_stock_data(ticker, period=f"{days}d")
        if df is not None and not df.empty:
            return df
    
    # Fall back to local data if real-time fails or is disabled
    return get_local_stock_history(ticker, days)

# Function to predict stock prices with real-time data
def predict_stock_prices(ticker, days_to_predict=7):
    try:
        # Get historical data (always use real-time for predictions)
        df = get_stock_history(ticker, days=60, use_realtime=True)  # Get more data for better prediction
        
        if df is None or df.empty:
            return None, None
        
        # Sort by date ascending for prediction
        df = df.sort_values('date')
        
        # Create features (X) and target (y)
        df['day_number'] = range(len(df))
        X = df[['day_number']]
        y = df['close']
        
        # Create and train the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict for future days
        last_day = df['day_number'].iloc[-1]
        future_days = np.array(range(last_day + 1, last_day + days_to_predict + 1))
        future_days = future_days.reshape(-1, 1)
        
        # Make predictions
        predictions = model.predict(future_days)
        
        # Create dates for predictions
        last_date = df['date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'date': future_dates,
            'predicted_close': predictions
        })
        
        # Calculate trend
        first_pred = predictions[0]
        last_pred = predictions[-1]
        trend_pct = ((last_pred - first_pred) / first_pred) * 100
        
        # Add current price and timestamp for real-time display
        pred_df['current_price'] = df['close'].iloc[-1]
        pred_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return pred_df, trend_pct
    
    except Exception as e:
        st.error(f"Error predicting stock prices: {e}")
        return None, None

# Load or create sample data
try:
    stock_data = pd.read_csv("pre-process/stocks_dataset/stock_name_ticker.csv", on_bad_lines='skip')
except Exception:
    # Create a minimal dataset if file not found
    stock_data = pd.DataFrame({
        "company_name": ["Apple", "Microsoft", "Google", "Amazon", "Tesla"],
        "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    })
    # Save it for future use
    os.makedirs("pre-process/stocks_dataset", exist_ok=True)
    stock_data.to_csv("pre-process/stocks_dataset/stock_name_ticker.csv", index=False)

# Get unique company names
company_names = stock_data["company_name"].unique()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Stock Agent. Select a company and ask me anything about it."}]

# App title and description
st.title("Stock Agent")
st.markdown("Ask questions about stocks and get instant answers!")

# Sidebar for company selection
with st.sidebar:
    st.header("Company Selection")
    selected_company = st.selectbox("Choose a company:", options=[""] + list(company_names))
    
    if st.button("Reset Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Stock Agent. Select a company and ask me anything about it."}]
        st.experimental_rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Display stock data if available
if selected_company:
    # Get ticker from company name
    ticker = stock_data[stock_data["company_name"] == selected_company]["ticker"].iloc[0]
    
    if ticker:
        st.sidebar.write(f"Ticker: {ticker}")
        stock_data_history = get_stock_history(ticker)
        if stock_data_history is not None:
            st.sidebar.subheader("Recent Stock Prices")
            st.sidebar.dataframe(stock_data_history[["date", "close"]].head())
            
            # Create a simple chart
            st.sidebar.subheader("Price Chart")
            chart_data = stock_data_history.sort_values("date")
            st.sidebar.line_chart(chart_data.set_index("date")["close"])
            
            # Add price predictions
            st.sidebar.subheader("Price Predictions (Next 7 Days)")
            predictions, trend = predict_stock_prices(ticker)
            if predictions is not None:
                # Display prediction chart
                fig, ax = plt.figure(figsize=(10, 4)), plt.gca()
                ax.plot(predictions['date'], predictions['predicted_close'])
                ax.set_title(f"{selected_company} Price Prediction")
                ax.set_xlabel("Date")
                ax.set_ylabel("Predicted Price ($)")
                plt.xticks(rotation=45)
                st.sidebar.pyplot(fig)
                
                # Display trend information
                trend_direction = "up" if trend > 0 else "down"
                st.sidebar.info(f"Predicted trend: Price will likely go {trend_direction} by {abs(trend):.2f}% in the next 7 days")

# Chat input
if prompt := st.chat_input("Ask about a stock..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        if not selected_company:
            response = "Please select a company from the sidebar first."
        else:
            ticker = stock_data[stock_data["company_name"] == selected_company]["ticker"].iloc[0]
            
            # Enhanced response with stock data if available
            if "price" in prompt.lower() or "stock" in prompt.lower() or "market" in prompt.lower() or "predict" in prompt.lower():
                response = f"Based on real-time data for {selected_company} ({ticker}):\n\n"
                
                stock_data_history = get_stock_history(ticker)
                if stock_data_history is not None:
                    latest = stock_data_history.iloc[0]
                    prev = stock_data_history.iloc[1] if len(stock_data_history) > 1 else None
                    
                    response += f"Latest closing price (on {latest['date']}): ${latest['close']:.2f}\n"
                    if prev is not None:
                        change = latest['close'] - prev['close']
                        pct_change = (change / prev['close']) * 100
                        direction = "up" if change > 0 else "down"
                        response += f"Price moved {direction} ${abs(change):.2f} ({abs(pct_change):.2f}%) from previous day.\n"
                    
                    # Add prediction information if requested
                    if "predict" in prompt.lower() or "upcoming" in prompt.lower() or "next" in prompt.lower() or "future" in prompt.lower() or "forecast" in prompt.lower():
                        predictions, trend = predict_stock_prices(ticker)
                        if predictions is not None:
                            trend_direction = "up" if trend > 0 else "down"
                            response += f"\nMarket Prediction for Next 7 Days:\nBased on historical data analysis, {selected_company} stock is predicted to trend {trend_direction} by approximately {abs(trend):.2f}% over the next 7 days.\n\nPredicted prices:\n"
                            for i, row in predictions.head(7).iterrows():
                                response += f"- {row['date'].strftime('%Y-%m-%d')}: ${row['predicted_close']:.2f}\n"
                else:
                    response += "I couldn't find recent price data for this company."
            else:
                response = f"You asked about {selected_company} ({ticker}): {prompt}\n\nPlease ask about stock prices, market trends, or predictions for your selected company."
        
        st.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
