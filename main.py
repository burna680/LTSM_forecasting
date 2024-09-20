#######################################################################################################################
########################################################### IMPORTS ###################################################
#######################################################################################################################

import streamlit as st
from streamlit_option_menu import option_menu

from src.plotting import *
from src.data_handling import *
from src.constants import *
from src.create_model import *
from src.pages.HomePage import HomePage
from src.pages.ContactPage import ContactPage
from src.pages.DataGatheringPage import DataGathering
from src.pages.DataAnalysisPage import DataAnalysis
from src.pages.DataPreprocessingPage import DataPreprocessing
from src.pages.ModelPage import ModelPage
from src.pages.ResultsPage import ResultsPage

#######################################################################################################################
########################################################### CACHE FUNCTIONS ###########################################
#######################################################################################################################

gather_data=st.cache_data(gather_data)
plot_stats=st.cache_data(plot_stats)
LSTMForecasting= st.cache_resource(LSTMForecasting)
#######################################################################################################################
########################################################## MAIN STREAMLIT SECTION #####################################
#######################################################################################################################

if __name__ == "__main__":
    pages = {
        "Home": HomePage("Home"),
        "Data gathering": DataGathering("Data gathering"),
        "Data analysis": DataAnalysis("Data analysis"),
        "Data preprocessing": DataPreprocessing("Data preprocessing"),
        "Model training": ModelPage("Model training"),
        "Results": ResultsPage("Results"),
        "Contact Me": ContactPage("Contact Me")
    }
    st.set_page_config(
        page_title="LTSM Forecasting App",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    #Icons for the side menu https://icons.getbootstrap.com/

    with st.sidebar:
        st.image("forecasting_LTSM.png", width=300)
        # selected_page = st.selectbox("Main Menu", list(pages.keys()))
        selected = option_menu(
            menu_title = "Main Menu",
            options = ["Home",
                       "Data gathering",
                       "Data analysis",
                       "Data preprocessing",
                       "Model training",
                       "Results",
                       "Contact Me"],
            icons = ["house",
                     "database-fill-down",
                     "activity",
                     "clipboard-data-fill",
                     "gear-fill",
                     "bar-chart-fill",
                     "envelope"],
            menu_icon = "cast",
            default_index = 0,
            #orientation = "horizontal",
            )
    try:
        if selected in pages:
            pages[selected].render()
    except Exception as e:
        st.error(e)