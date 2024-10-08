import streamlit as st
from src.pages.Page import Page
from src.create_model import *
from src.train_model import *
from src.plotting import *
from src.predictions import *

class ModelPage(Page):
    def render(self):
        """
        Renders the Model page, which allows users to train an LSTM model using the preprocessed data
        from the Data Preprocessing page.

        Raises:
        - Exception: If no data is available (i.e. the user has not gathered data from the Data Gathering page).
        - Exception: If preprocessing is incomplete (i.e. the user has not completed Data Preprocessing).
        - Exception: If an error occurs while training the model.
        """
        st.subheader("Model Training")
        
        # Debugging statement to view session state
        # st.write("Current Session State:", st.session_state)

        # Check if data is available
        if "stock_data" not in st.session_state or not st.session_state["stock_data"]:
            raise Exception("No data available. Please gather data first from the 'Data Gathering' page.")
        # st.write("Current Session State:", st.session_state)
        # Check if preprocessing is done
        required_keys = ["window_size", "feature_selected", "X_train", "y_train"]
        missing_keys = [key for key in required_keys if st.session_state.get(key) is None]
        if missing_keys:
            raise Exception(f"Missing preprocessing data: {', '.join(missing_keys)}. Please complete Data Preprocessing first.") from e 
        left_col, right_col = st.columns(2)
        with left_col:
            with st.container():
                lstm_layers = st.slider('LSTM Layers:', 1, 5, value=2, key="lstm_layers")
                lstm_neurons = st.slider('LSTM Neurons:', 1, 50,  value=50, key="lstm_neurons")
                st.divider()
                linear_hidden_layers = st.slider('Hidden Layers:', 1, 5,  value=2, key="linear_hidden_layers")
                linear_hidden_neurons = st.slider('Hidden Neurons:', 1, 50,  value=10, key="linear_hidden_neurons")
        with right_col:
            with st.container():
                n_epochs = st.number_input('No. of Epochs', step=1, min_value=1,  value=150, max_value=500, key="n_epochs")
                batch_size = st.number_input('Batch Size', step=1, min_value=1, value=16, max_value=100, key="batch_size")
                learning_rate = st.number_input('Learning Rate', value=0.001,  step=0.0001, min_value=0.000001, max_value=1.0, key="learning_rate",format="%0.4f")
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
                raise Exception("Error training the model") from e
