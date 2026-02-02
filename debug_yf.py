import yfinance as yf

tickers = ['PETR4.SA', 'VALE3.SA']
data = yf.download(tickers, start='2024-01-01', end='2024-01-10')
print("Columns:", data.columns)
print("Head:\n", data.head())
