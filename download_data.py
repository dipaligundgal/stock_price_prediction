import yfinance as yf
import pandas as pd

# Download stock data
stock = yf.download(
    "AAPL",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=True
)

# Convert MultiIndex columns to single-level
if isinstance(stock.columns, pd.MultiIndex):
    stock.columns = stock.columns.get_level_values(0)

# Save CSV
stock.to_csv("data/apple.csv")

print(stock.head())