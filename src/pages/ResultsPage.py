import streamlit as st
from src.pages.Page import Page
from src.plotting import *
from src.predictions import *

class ResultsPage(Page):
    def render(self):
        st.subheader("Results")
        
        try: 
            history=st.session_state.history
            model=st.session_state.model
            st.plotly_chart(plot_training_performance(history))
            # Predict stock price for all data splits
            train_results, val_results, test_results = predict(
                model=  model,
                X_train=st.session_state.X_train,
                X_validate=st.session_state.X_val,
                X_test=st.session_state.X_test,
                y_train=st.session_state.y_train,
                y_validate=st.session_state.y_val,
                y_test=st.session_state.y_test,
                scaler=st.session_state.scaler
            )

            actual_data = st.session_state.splitted_data

            predicted_data = {
                "Train": train_results,  
                "Validation": val_results,
                "Test": test_results
            }

            colors = {
                "Train": "lightblue",
                "Validation": "orange",
                "Test": "darkred"
            }
            st.write(train_results)
            st.plotly_chart(plot_actual_vs_predicted_stock_data(actual_data, predicted_data, colors))
            # st.plotly_chart(fig)
        except Exception as e:
            raise TypeError(e) from e