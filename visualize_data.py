import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("data/apple.csv")

# Create a graph
plt.figure(figsize=(12,6))

plt.plot(data["Close"], color="blue", linewidth=2)

plt.title("Apple Stock Closing Price")
plt.xlabel("Trading Days")
plt.ylabel("Closing Price (USD)")

plt.grid(True)

plt.show()