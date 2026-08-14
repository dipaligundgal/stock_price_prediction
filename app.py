import streamlit as st
import pickle

# Load the trained model
with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("📈 Stock Price Prediction")

st.write("Enter the stock details below:")

open_price = st.number_input("Open Price")
high_price = st.number_input("High Price")
low_price = st.number_input("Low Price")
volume = st.number_input("Volume", min_value=0)

if st.button("Predict Closing Price"):
    prediction = model.predict([[open_price, high_price, low_price, volume]])

    st.success(f"Predicted Closing Price: ${prediction[0]:.2f}")