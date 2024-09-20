import numpy as np

def predict(model, X_train, X_validate, X_test, y_train, y_validate, y_test, scaler):
    """
    Predicts and inversely transforms the predicted and actual data.
    
    Args:
    - model: Trained model for making predictions.
    - X_train: Training data features.
    - X_validate: Validation data features.
    - X_test: Testing data features.
    - y_train: Actual training data labels.
    - y_validate: Actual validation data labels.
    - y_test: Actual testing data labels.
    - scaler: Scaler used for inverse transformation (e.g., StandardScaler or MinMaxScaler).

    Returns:
    - A tuple containing the inversely transformed predictions and actual data for training, validation, and test sets.
    """
    # Make predictions
    y_train_predict = model.predict(X_train)
    y_validate_predict = model.predict(X_validate)
    y_test_predict = model.predict(X_test)

    # Inverse transform actual data
    y_train_inv = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), np.ones((len(y_train.reshape(-1, 1)), 5))), axis=1))[:, 0]
    y_validate_inv = scaler.inverse_transform(np.concatenate((y_validate.reshape(-1, 1), np.ones((len(y_validate.reshape(-1, 1)), 5))), axis=1))[:, 0]
    y_test_inv = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.ones((len(y_test.reshape(-1, 1)), 5))), axis=1))[:, 0]

    # Inverse transform predicted data
    y_train_predict_inv = scaler.inverse_transform(np.concatenate((y_train_predict, np.ones((len(y_train_predict), 5))), axis=1))[:, 0]
    y_validate_predict_inv = scaler.inverse_transform(np.concatenate((y_validate_predict, np.ones((len(y_validate_predict), 5))), axis=1))[:, 0]
    y_test_predict_inv = scaler.inverse_transform(np.concatenate((y_test_predict, np.ones((len(y_test_predict), 5))), axis=1))[:, 0]

    return y_train_predict_inv, y_validate_predict_inv, y_test_predict_inv
