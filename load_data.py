import yfinance as yf
import pandas as pd
import os

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(data_dir, exist_ok=True)

def fetch_and_store(ticker, period="1y"):
    """Fetch stock history from Yahoo Finance and save to CSV"""
    try:
        # Fetch stock history
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        # Reset index to make date a column
        hist = hist.reset_index()
        
        # Convert date to string format
        hist['Date'] = hist['Date'].dt.date
        
        # Add ticker column
        hist['ticker'] = ticker
        
        # Save to CSV
        csv_path = os.path.join(data_dir, f"{ticker}_history.csv")
        hist.to_csv(csv_path, index=False)
        
        print(f"✅ Data stored for {ticker} ({len(hist)} records) in {csv_path}")
        return hist
    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return None

if __name__ == "__main__":
    # Load data for common stocks
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    all_data = []
    
    for ticker in tickers:
        data = fetch_and_store(ticker, period="1y")
        if data is not None:
            all_data.append(data)
    
    # Combine all data into a single file
    if all_data:
        combined = pd.concat(all_data)
        combined_path = os.path.join(data_dir, "all_stocks_history.csv")
        combined.to_csv(combined_path, index=False)
        print(f"✅ Combined data saved to {combined_path}")