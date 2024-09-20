import streamlit as st
from src.pages.Page import Page
from src.create_model import *
from src.train_model import *
from src.plotting import *
from src.predictions import *

class ModelPage(Page):
    def render(self):
        st.subheader("Model Training")
        
        # Debugging statement to view session state
        # st.write("Current Session State:", st.session_state)

        # Check if data is available
        if "stock_data" not in st.session_state or not st.session_state["stock_data"]:
            st.error("No data available. Please gather data first from the 'Data Gathering' page.")
            return
        # st.write("Current Session State:", st.session_state)
        # Check if preprocessing is done
        required_keys = ["window_size", "feature_selected", "X_train", "y_train"]
        missing_keys = [key for key in required_keys if st.session_state.get(key) is None]
        if missing_keys:
            st.error(f"Missing preprocessing data: {', '.join(missing_keys)}. Please complete Data Preprocessing first.")
            return

        left_col, right_col = st.columns(2)
        with left_col:
            with st.container():
                lstm_layers = st.slider('LSTM Layers:', 1, 5, key="lstm_layers")
                lstm_neurons = st.slider('LSTM Neurons:', 1, 50, key="lstm_neurons")
                st.divider()
                linear_hidden_layers = st.slider('Hidden Layers:', 1, 5, key="linear_hidden_layers")
                linear_hidden_neurons = st.slider('Hidden Neurons:', 1, 50, key="linear_hidden_neurons")
        with right_col:
            with st.container():
                n_epochs = st.number_input('No. of Epochs', step=1, min_value=1, max_value=500, key="n_epochs")
                batch_size = st.number_input('Batch Size', step=1, min_value=1, max_value=100, key="batch_size")
                learning_rate = st.number_input('Learning Rate', value=0.001,  step=0.0001, min_value=0.0001, max_value=1.0, key="learning_rate")
            # Button to train the model
        train_click = st.button('Train LSTM Model', type="primary")
        if train_click:
            try:
                model = LSTMForecasting(
                    input_size=st.session_state.window_size,
                    lstm_hidden_size=st.session_state.lstm_neurons,
                    lstm_num_layers=st.session_state.lstm_layers,
                    linear_num_layers=st.session_state.linear_hidden_layers,
                    linear_hidden_size=st.session_state.linear_hidden_neurons,
                    output_size=1
                )                 
                model, history= train_model(model, 
                                            learning_rate=st.session_state.learning_rate,
                                            epochs=st.session_state.n_epochs,
                                            batch_size=st.session_state.batch_size, 
                                            X_train=st.session_state.X_train,
                                            y_train=st.session_state.y_train, 
                                            X_val=st.session_state.X_val,
                                            y_val=st.session_state.y_val)  
                st.session_state.model = model
                st.session_state.history = history
            except Exception as e:
                st.error(f"Error training the model: {e}")
        # st.error('Adjusting features or parameters after training will not maintain the session. Please ensure to retrain after making changes.', icon="ℹ️")