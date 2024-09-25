import streamlit as st
from src.pages.Page import Page
from src.data_handling import *
from src.plotting import *

class DataPreprocessing(Page):
    def render(self):
        st.subheader("Data Preprocessing")

        if "stock_data" not in st.session_state:
            raise Exception("No data available. Please gather data first from the 'Data Gathering' page.")
        if 'window_size' not in st.session_state:
            st.session_state.window_size = 8  # Default value as an integer
        data = st.session_state.stock_data
        stock_selected = list(st.session_state.stock_data.keys())[0]
        st.session_state.stock_selected = stock_selected
        
        left_col, right_col = st.columns(2)
        with right_col:
            window_size_input = st.text_input(
                "Window size",
                value=8,
                help="Time frame to use for prediction"
            )

        with left_col:
            feature_selected = st.selectbox(
                "Choose a feature to predict",
                options=data[stock_selected].columns,
                # key="feature_selected"
            )

        try:
            # Convert window_size to integer and handle possible conversion errors
            window_size = int(window_size_input)
            st.session_state.window_size = window_size  # Ensure it's stored as an integer
            st.session_state.feature_selected = feature_selected
            train_data, val_data, test_data = split_dataset_sequentially(
                data,
                [stock_selected, st.session_state.feature_selected],
                train_size=0.7,
                val_size=0.1,
                test_size=0.2
            )
            splitted_data = {
                "Train": train_data,
                "Validation": val_data,
                "Test": test_data
            }
            st.session_state.splitted_data = splitted_data

            colors = {
                "Train": "lightblue",
                "Validation": "orange",
                "Test": "darkred"
            }
            st.pyplot(plot_splitted_stock_data(splitted_data, colors))

            # Preprocess the data
            X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_data(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                seq_size=window_size
            )
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_val = X_val
            st.session_state.y_val = y_val
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler

        except ValueError as ve:
            raise ValueError("Value Error during preprocessing") from ve
        except Exception as e:
            raise Exception("Error preprocessing data") from e 