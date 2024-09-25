import streamlit as st
from src.pages.Page import Page
from src.plotting import *
class DataAnalysis(Page):
    def render(self):
        st.header("Data Analysis")
        if "stock_data" not in st.session_state:
            raise Exception("No data available. Please gather data first from the 'Data Gathering' page.")
        stock_selected = list(st.session_state.stock_data.keys())[0]
        data = st.session_state.stock_data[stock_selected]
        stats = data['Close'].agg(['mean', 'std', 'min', 'max'])
        missing_values = data['Close'].isnull().sum()
        left_col, right_col = st.columns(2)
        with left_col:
            st.write("Analyzing stock data for:", stock_selected)
            st.write("Mean: ", stats['mean'])
            st.write("STD: ", stats['std'])
        with right_col:
            st.write("Min: ", stats['min'])
            st.write("Max: ", stats['max'])
            st.write("Missing values: ", missing_values)

        fig_candlestick, fig = st.session_state.get("figs", (None, None))
        if fig_candlestick is None or fig is None or st.session_state.stock_data_key != stock_selected:
            try:
                fig_candlestick, fig = plot_stats(st.session_state.stock_data, stock_selected)
                st.session_state.figs = (fig_candlestick, fig)
                st.session_state.stock_data_key = stock_selected
                st.pyplot(fig_candlestick)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                raise Exception(f"Error analyzing stock data") from e
        else:
            st.pyplot(fig_candlestick)
            st.plotly_chart(fig, use_container_width=True)
