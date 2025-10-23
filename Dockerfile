FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Streamlit and other essential packages explicitly
RUN pip install --no-cache-dir streamlit pandas
RUN pip install --no-cache-dir streamlit-chat || echo "streamlit-chat not available, using fallback"
RUN pip install --no-cache-dir -r requirements.txt || echo "Some requirements may not be installed"

# Copy application code
COPY . .

# Create directory for CSV file if it doesn't exist
RUN mkdir -p pre-process/stocks_dataset

# Ensure the CSV file exists
COPY pre-process/stocks_dataset/stock_name_ticker.csv pre-process/stocks_dataset/ || echo "CSV file not copied, will use fallback data"

# Set environment variables for database connections
ENV MYSQL_HOST=host.docker.internal
ENV MYSQL_PORT=3307
ENV NEO4J_URI=bolt://host.docker.internal:7687
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]