from src.data_handling import split_dataset_sequentially

import pytest
import pandas as pd

def test_valid_input():
    data = {'stock1': pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})}
    target = ['stock1', 'feature1']
    train_size = 0.7
    val_size = 0.1
    test_size = 0.2

    train_data, val_data, test_data = split_dataset_sequentially(data, target, train_size, val_size, test_size)

    assert isinstance(train_data, pd.DataFrame)
    assert isinstance(val_data, pd.DataFrame)
    assert isinstance(test_data, pd.DataFrame)

    assert len(train_data) == 3
    assert len(val_data) == 1
    assert len(test_data) == 1

def test_invalid_input_data():
    data = 'not a dict'
    target = ['stock1', 'feature1']
    train_size = 0.7
    val_size = 0.1
    test_size = 0.2

    with pytest.raises(AssertionError):
        split_dataset_sequentially(data, target, train_size, val_size, test_size)

def test_invalid_target_stock_or_feature():
    data = {'stock1': pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})}
    target = ['stock2', 'feature1']  # invalid stock
    train_size = 0.7
    val_size = 0.1
    test_size = 0.2

    with pytest.raises(KeyError):
        split_dataset_sequentially(data, target, train_size, val_size, test_size)

def test_invalid_train_val_test_size_proportions():
    data = {'stock1': pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})}
    target = ['stock1', 'feature1']
    train_size = 0.8
    val_size = 0.2
    test_size = 0.1  # invalid proportions

    with pytest.raises(AssertionError):
        split_dataset_sequentially(data, target, train_size, val_size, test_size)

def test_edge_cases():
    data = {'stock1': pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})}
    target = ['stock1', 'feature1']
    train_size = 1.0
    val_size = 0.0
    test_size = 0.0

    train_data, val_data, test_data = split_dataset_sequentially(data, target, train_size, val_size, test_size)

    assert isinstance(train_data, pd.DataFrame)
    assert isinstance(val_data, pd.DataFrame)
    assert isinstance(test_data, pd.DataFrame)

    assert len(train_data) == 5
    assert len(val_data) == 0
    assert len(test_data) == 0

def test_empty_data():
    data = {}
    target = ['stock1', 'feature1']
    train_size = 0.7
    val_size = 0.1
    test_size = 0.2

    with pytest.raises(KeyError):
        split_dataset_sequentially(data, target, train_size, val_size, test_size)

def test_empty_target_stock():
    data = {'stock1': pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})}
    target = ['', 'feature1']
    train_size = 0.7
    val_size = 0.1
    test_size = 0.2

    with pytest.raises(KeyError):
        split_dataset_sequentially(data, target, train_size, val_size, test_size)

def test_empty_target_feature():
    data = {'stock1': pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})}
    target = ['stock1', '']
    train_size = 0.7
    val_size = 0.1
    test_size = 0.2

    with pytest.raises(KeyError):
        split_dataset_sequentially(data, target, train_size, val_size, test_size)

def test_negative_train_val_test_size():
    data = {'stock1': pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})}
    target = ['stock1', 'feature1']
    train_size = -0.1
    val_size = 0.2
    test_size = 0.7

    with pytest.raises(AssertionError):
        split_dataset_sequentially(data, target, train_size, val_size, test_size)

def test_train_val_test_size_sum_greater_than_one():
    data = {'stock1': pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})}
    target = ['stock1', 'feature1']
    train_size = 0.6
    val_size = 0.3
    test_size = 0.2

    with pytest.raises(AssertionError):
        split_dataset_sequentially(data, target, train_size, val_size, test_size)