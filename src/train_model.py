from keras import optimizers
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Sequential
from keras.models import load_model

import numpy as np
import streamlit as st
# Custom callback to update Streamlit during training
class StreamlitProgressCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.epoch_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_counter += 1
        progress_percentage = int((self.epoch_counter / self.total_epochs) * 100)
        self.progress_bar.progress(progress_percentage)
        self.status_text.text(f"Epoch {self.epoch_counter}/{self.total_epochs} - Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")


def train_model(
            model: Sequential,
            learning_rate: float = 0.01,
            epochs: int = 100,
            batch_size: int =16,
            X_train: np.ndarray = None,
            y_train: np.ndarray = None,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None
    ) -> tuple[Sequential, dict]:
    """
    Trains a model using the given parameters.

    Args:
    - model: The model to be trained
    - learning_rate: The learning rate for the optimizer
    - epochs: The number of epochs to train for
    - X_train: The input data for training
    - y_train: The labels for the training data
    - X_val: The input data for validation
    - y_val: The labels for the validation data

    Returns:
    - The trained model
    - A dictionary of the training history
    """
    if model is None:
        raise ValueError("Model cannot be null")
    if X_train is None or y_train is None:
        raise ValueError("Training data cannot be null")
    if X_val is None or y_val is None:
        raise ValueError("Validation data cannot be null")

    # Define loss function, metrics, and optimizer
    loss_fn = "mean_squared_error"
    # metrics_list = ["mean_absolute_error"]
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn)#, metrics=metrics_list)

    # Set model checkpoint path
    model_location = "./models/"
    model_name = "model.keras"
    best_model_checkpoint_callback = ModelCheckpoint(
        model_location + model_name,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=0
    )
    streamlit_progress_callback = StreamlitProgressCallback(epochs)


    # Train the model
    try:
        history = model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[best_model_checkpoint_callback,streamlit_progress_callback]
        )
    except Exception as e:
        raise RuntimeError(f"Error occurred during model training: {str(e)}")
    # Load the best performing model
    best_model = load_model(model_location + model_name)
    return best_model, history.history
