from src.pages.Page import Page
import streamlit as st
class HomePage(Page):
    def render(self):
        st.header("LTSM Forecasting App")
