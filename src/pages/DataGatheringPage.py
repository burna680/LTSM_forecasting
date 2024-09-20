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
        st.selectbox("Choose a stock", df["symbol"], key="stock_selected")
        # Gather stock data when a stock is selected
        if st.session_state.stock_selected:
            # Fetch stock data every time the user selects a new stock
            try:
                st.session_state.stock_data = gather_data(stocks=[st.session_state.stock_selected])
                st.write(st.session_state.stock_data[st.session_state.stock_selected])
            except Exception as e:
                st.error(f"Error displaying stock data: {e}")
