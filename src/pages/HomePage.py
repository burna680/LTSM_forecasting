from src.pages.Page import Page
import streamlit as st
class HomePage(Page):
    def render(self):
        st.header("LTSM Forecasting App")
        st.subheader("Welcome to the Long Short-Term Memory (LSTM) stock price forecasting app!")
        st.write("This app allows you to gather historical stock price data, analyze and preprocess it, train an LSTM model, and view the forecasted results.")
        st.image("misc/stocks.gif", width=500)
        st.write("Get started by following these steps:")
        st.write("**Usage Guide**")
        st.write("1. **Gather Data**: Use the app to gather historical stock price data for a specific stock ticker.")
        st.write("2. **Analyze and Preprocess Data**: The app will analyze and preprocess the data for training the LSTM model.")
        st.write("3. **Train Model**: Train the LSTM model using the preprocessed data.")
        st.write("4. **View Results**: View the forecasted stock prices and compare them to the actual prices.")
        st.write("Select a page from the sidebar menu to begin!")
        