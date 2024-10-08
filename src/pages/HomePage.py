from src.pages.Page import Page
import streamlit as st
class HomePage(Page):
    def render(self):
        st.header("LTSM Forecasting App")
        st.subheader("Welcome to the Long Short-Term Memory (LSTM) stock price forecasting app!")
        st.write("This app allows you to gather historical stock price data, analyze and preprocess it, train an LSTM model, and view the forecasted results.")
        left_col, right_col = st.columns(2)
        with right_col:
            st.image("misc/stocks.gif", width=400)
        with left_col:
            st.title("**User Guide**")
            st.write("1. Go to the **Gather Data** page and use the app to gather historical stock price data for a specific stock ticker.")
            st.write("2. You can see stast in the **Data analysis** page")
            st.write("3. See the preprocessing step in the **Data preprocessing** page to see the data for training the LSTM model.")
            st.write("4. **Train Model**: Train the LSTM model using the preprocessed data.")
            st.write("5. **View Results**: View the forecasted stock prices and compare them to the actual prices.")        