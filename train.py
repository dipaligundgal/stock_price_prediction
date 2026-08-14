import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("data/apple.csv")

# Display first 5 rows
print("First 5 Rows:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# -------------------------
# Feature Selection
# -------------------------

# Input features
X = data[["Open", "High", "Low", "Volume"]]

# Output (Target)
y = data["Close"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)


# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(predictions[:10])


print("\nActual Close Prices:")
print(y_test.head(10))

print("\nPredicted Close Prices:")
print(predictions[:10])

# Evaluate the model

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("----------------------")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)


import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(y_test.values, label="Actual Price", color="blue")
plt.plot(predictions, label="Predicted Price", color="red")

plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Test Data")
plt.ylabel("Closing Price")

plt.legend()
plt.grid(True)

plt.show()

import os
import pickle

# Create the models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the trained model
with open("models/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")