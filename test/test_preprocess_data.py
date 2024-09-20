import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.data_handling import preprocess_data

def test_valid_input():
    # Create sample data and convert to DataFrame to ensure 2D input
    train_data = pd.DataFrame(np.random.rand(100), columns=["value"])
    val_data = pd.DataFrame(np.random.rand(50), columns=["value"])
    test_data = pd.DataFrame(np.random.rand(50), columns=["value"])
    seq_size = 8  # Define the sequence size for the test

    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(train_data, val_data, test_data, seq_size=seq_size)

    # Check shapes
    assert X_train.shape == (100 - seq_size, seq_size, 1)
    assert y_train.shape == (100 - seq_size,)
    assert X_val.shape == (50 - seq_size, seq_size, 1)
    assert y_val.shape == (50 - seq_size,)
    assert X_test.shape == (50 - seq_size, seq_size, 1)
    assert y_test.shape == (50 - seq_size,)

def test_invalid_input():
    # Create sample data with non-numeric values
    train_data = pd.DataFrame(['a', 'b', 'c'], columns=["value"])
    val_data = pd.DataFrame([1, 2, 3], columns=["value"])
    test_data = pd.DataFrame([4, 5, 6], columns=["value"])

    # Check that an error is raised
    with pytest.raises(ValueError):
        preprocess_data(train_data, val_data, test_data, seq_size=3)

def test_empty_input():
    # Create empty data
    train_data = pd.DataFrame([], columns=["value"])
    val_data = pd.DataFrame([], columns=["value"])
    test_data = pd.DataFrame([], columns=["value"])

    # Check that an error is raised
    with pytest.raises(ValueError):
        preprocess_data(train_data, val_data, test_data, seq_size=3)

def test_input_data_of_different_lengths():
    # Create sample data of different lengths
    train_data = pd.DataFrame(np.random.rand(100), columns=["value"])
    val_data = pd.DataFrame(np.random.rand(50), columns=["value"])
    test_data = pd.DataFrame(np.random.rand(200), columns=["value"])
    seq_size = 8  # Define sequence size

    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(train_data, val_data, test_data, seq_size=seq_size)

    # Check shapes
    assert X_train.shape == (100 - seq_size, seq_size, 1)
    assert y_train.shape == (100 - seq_size,)
    assert X_val.shape == (50 - seq_size, seq_size, 1)
    assert y_val.shape == (50 - seq_size,)
    assert X_test.shape == (200 - seq_size, seq_size, 1)
    assert y_test.shape == (200 - seq_size,)
def test_input_data_of_different_shapes():
    # Create sample data of different shapes (multivariate for both train, val, and test)
    train_data = pd.DataFrame(np.random.rand(100, 2), columns=["value1", "value2"])
    val_data = pd.DataFrame(np.random.rand(50, 2), columns=["value1", "value2"])
    test_data = pd.DataFrame(np.random.rand(200, 2), columns=["value1", "value2"])
    seq_size = 8  # Define sequence size

    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(train_data, val_data, test_data, seq_size=seq_size)

    # Check shapes
    assert X_train.shape == (100 - seq_size, seq_size, 2)
    assert y_train.shape == (100 - seq_size, 2)
    assert X_val.shape == (50 - seq_size, seq_size, 2)
    assert y_val.shape == (50 - seq_size, 2)
    assert X_test.shape == (200 - seq_size, seq_size, 2)
    assert y_test.shape == (200 - seq_size, 2)

def test_preprocess_data_with_missing_values():
    # Create sample data with missing values
    train_data = pd.DataFrame([1, 2, np.nan, 4, 5], columns=["value"])
    val_data = pd.DataFrame([6, 7, 8, 9, 10], columns=["value"])
    test_data = pd.DataFrame([11, 12, 13, 14, 15], columns=["value"])
    seq_size = 3  # Define sequence size

    # Check that an error is raised
    with pytest.raises(ValueError):
        preprocess_data(train_data, val_data, test_data, seq_size=seq_size)

def test_preprocess_data_with_infinite_values():
    # Create sample data with infinite values
    train_data = pd.DataFrame([1, 2, np.inf, 4, 5], columns=["value"])
    val_data = pd.DataFrame([6, 7, 8, 9, 10], columns=["value"])
    test_data = pd.DataFrame([11, 12, 13, 14, 15], columns=["value"])
    seq_size = 3  # Define sequence size

    # Check that an error is raised
    with pytest.raises(ValueError):
        preprocess_data(train_data, val_data, test_data, seq_size=seq_size)
