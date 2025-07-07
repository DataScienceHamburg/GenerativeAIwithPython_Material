# filename: plot_eth_sol_ytd.py
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

# Define the cryptocurrencies and the start date for YTD
cryptos = ['ETH-USD', 'SOL-USD']
start_date = datetime.datetime(datetime.datetime.now().year, 1, 1)

# Fetch the historical data
data = yf.download(cryptos, start=start_date)

# Calculate the YTD price change
ytd_prices = data['Close']
ytd_price_change = (ytd_prices.iloc[-1] - ytd_prices.iloc[0]) / ytd_prices.iloc[0] * 100

# Plotting
plt.figure(figsize=(10, 5))
ytd_price_change.plot(kind='bar', color=['blue', 'orange'])
plt.title('YTD Price Change of ETH and SOL')
plt.ylabel('Percentage Change (%)')
plt.xlabel('Cryptocurrency')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(axis='y')

# Show the plot
plt.tight_layout()
plt.show()