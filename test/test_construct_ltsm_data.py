from src.data_handling import construct_lstm_data
import numpy as np
import pytest
def test_construct_lstm_data_with_sequence_size_8():
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]).T
    expected_data_X = np.array([[[1, 11], [2, 12], [3, 13], [4, 14], [5, 15], [6, 16], [7, 17], [8, 18]],
                                [[2, 12], [3, 13], [4, 14], [5, 15], [6, 16], [7, 17], [8, 18], [9, 19]]])
    expected_data_y = np.array([[9, 19], [10, 20]])
    
    data_X, data_y = construct_lstm_data(data, sequence_size=8)
    
    np.testing.assert_array_equal(data_X, expected_data_X)
    np.testing.assert_array_equal(data_y, expected_data_y)

def test_construct_lstm_data_with_sequence_size_5():
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]).T  # Transposed to ensure it fits the shape
    expected_data_X = np.array([[[1, 11], [2, 12], [3, 13], [4, 14], [5, 15]],
                                [[2, 12], [3, 13], [4, 14], [5, 15], [6, 16]],
                                [[3, 13], [4, 14], [5, 15], [6, 16], [7, 17]],
                                [[4, 14], [5, 15], [6, 16], [7, 17], [8, 18]],
                                [[5, 15], [6, 16], [7, 17], [8, 18], [9, 19]]])
    expected_data_y = np.array([[6, 16], [7, 17], [8, 18], [9, 19], [10, 20]])
    data_X, data_y = construct_lstm_data(data, sequence_size=5)
    np.testing.assert_array_equal(data_X, expected_data_X)
    np.testing.assert_array_equal(data_y, expected_data_y)
def test_construct_lstm_data_with_sequence_size_of_1():
    data = np.array([[1, 2, 3, 4, 5]]).T
    expected_data_X = np.array([[[1]], [[2]], [[3]], [[4]]])
    expected_data_y = np.array([2, 3, 4, 5])
    
    data_X, data_y = construct_lstm_data(data, sequence_size=1)
    
    np.testing.assert_array_equal(data_X, expected_data_X)
    np.testing.assert_array_equal(data_y, expected_data_y)

def test_construct_lstm_data_with_input_data_of_shape_3():
    data = np.array([[[1, 2, 3], [4, 5, 6]], 
                     [[7, 8, 9], [10, 11, 12]], 
                     [[13, 14, 15], [16, 17, 18]]])
    expected_data_X = np.array([[[[1, 2, 3], [4, 5, 6]]], [[[7, 8, 9], [10, 11, 12]]]])
    expected_data_y = np.array([[[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]])
    
    data_X, data_y = construct_lstm_data(data, sequence_size=1)
    
    np.testing.assert_array_equal(data_X, expected_data_X)
    np.testing.assert_array_equal(data_y, expected_data_y)

def test_construct_lstm_data_with_invalid_sequence_size():
    data = np.array([[1, 2, 3, 4, 5]])
    with pytest.raises(ValueError):
        construct_lstm_data(data, sequence_size=-1)

def test_construct_lstm_data_with_non_numeric_input_data():
    data = np.array([['a', 'b', 'c'], ['d', 'e', 'f']])
    
    with pytest.raises(TypeError):
        construct_lstm_data(data, sequence_size=2)