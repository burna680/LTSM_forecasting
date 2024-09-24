import streamlit as st
from pytickersymbols import PyTickerSymbols
import time
import pandas as pd

from src.pages.Page import Page
from src.data_handling import *

class DataGathering(Page):
    def render(self):
        st.header("Data gathering")
        start_time = time.time()
        # Fetch all available stocks
        try:
            stock_data = PyTickerSymbols().get_all_stocks()
            st.text(f"Data from stocks fetched in {time.time() - start_time:.2f} seconds")
            df = pd.DataFrame(stock_data).drop(['isins', 'akas', 'metadata', 'wiki_name', 'symbols'], axis=1)
            st.write(df)
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            return
        stock_selected = st.selectbox("Choose a stock", df["symbol"], key="stock_selected")
        # Gather stock data when a stock is selected
        if stock_selected:
            # Fetch stock data every time the user selects a new stock
            try:
                stock_data = gather_data(stocks=[stock_selected])
                if stock_selected not in stock_data:
                    st.error(f"Error: {stock_selected} is not available in Yahoo Finance database.")
                else:
                    st.session_state.stock_data = stock_data
                    st.write(st.session_state.stock_data[stock_selected])
            except Exception as e:
                st.error(f"Error displaying stock data: {e}")
