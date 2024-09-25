import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import asyncio
from pytickersymbols import PyTickerSymbols


async def fetch_data():
    return PyTickerSymbols().get_all_stocks()

async def download_data(stock,period):
    return yf.download(stock, period=period)


def gather_data(
    stocks: list[str] = ["AAPL","TSLA", "NVDA", "AMZN", "GOOGL"],
    period: str = "1y"
    ) -> dict[str,pd.DataFrame]:
    """
    Downloads the stock data for given stocks and period. The data is
    returned as a pandas DataFrame.

    Args:
        stocks (List[str]): List of stock symbols. Defaults to
            ["AAPL"].
        period (str): The period for which to download the data. This
            should be a string in the format accepted by yfinance.
            Default is "2y".

    Returns:
        dict[ str,pd.DataFrame]: a dictionary with the stock name as key and its corresponding downloaded data as value.
    """
    if stocks is None:
        stocks = ["AAPL"]
    if period is None:
        period = "1y"

    if not isinstance(stocks, list):
        raise TypeError("Stocks must be a list of strings.")
    if not isinstance(period, str):
        raise TypeError("Period must be a string.")

    period = period.strip()

    try:
        data = {}
        for stock in stocks:
            stock_data = asyncio.run(download_data(stock, period))
            if stock_data.empty:
                raise ValueError(f"No data found for stock symbol: {stock}")
            data[stock] = stock_data
    except Exception as e:
        raise ValueError(f"Error downloading data: {e}")

    for stock, df in data.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df['change_percent'] = df['Close'].pct_change() * 100
    return data

def split_dataset_sequentially(data: dict[str,pd.DataFrame], target: list[str,str], train_size: float = 0.7, val_size: float = 0.1, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset sequentially into training, validation, and test subsets.
    """
    assert isinstance(data, dict), "Expected 'data' to be a dictionary"
    assert train_size + val_size + test_size == 1.0, "Sizes must sum up to 1.0"
    
    # Get the target data
    data = data[target[0]][target[1]]
    n_samples = len(data)
    
    # Calculate initial number of samples for each set
    train_end = int(train_size * n_samples)
    val_end = train_end + int(val_size * n_samples)
    
    # Adjust rounding errors by assigning remaining samples to validation and test sets
    remaining = n_samples - (train_end + (val_end - train_end))
    if remaining > 0:
        val_end += 1
    
    # Ensure test set gets the remainder
    train_data = data.iloc[:train_end].to_frame()
    val_data = data.iloc[train_end:val_end].to_frame()
    test_data = data.iloc[val_end:].to_frame()

    return train_data, val_data, test_data

def construct_lstm_data(data: np.ndarray, sequence_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct input data (X) and target data (y) for LSTM model from a numpy array.

    Parameters:
    -----------
    data : numpy.ndarray
        Input data array of shape (n_samples, n_features) or higher dimensions.

    sequence_size : int, default=8
        Number of previous time steps to use as input features for predicting the next time step.

    Returns:
    --------
    data_X : numpy.ndarray
        Array of LSTM input sequences of shape (n_samples - sequence_size, sequence_size, n_features).

    data_y : numpy.ndarray
        Corresponding target values for each input sequence.
    """
    # Check for non-numeric data
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Input data must be numeric.")
    
    n_samples = data.shape[0]

    # Adjusted validation for sequence_size
    if sequence_size <= 0 or sequence_size >= n_samples:
        raise ValueError("sequence_size must be positive and less than the number of samples in data.")

    data_X = []
    data_y = []

    # Constructing the sequences for LSTM input
    for i in range(n_samples - sequence_size):
        data_X.append(data[i:i + sequence_size])
        data_y.append(data[i + sequence_size])

    return np.array(data_X), np.squeeze(np.array(data_y))  # Sq

def preprocess_data(
    train_data: pd.DataFrame, 
    val_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    seq_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses the data for LSTM by normalizing and splitting into sequences.

    Args:
        train_data (pd.DataFrame): The training data.
        val_data (pd.DataFrame): The validation data.
        test_data (pd.DataFrame): The testing data.
        seq_size (int): The sequence size for LSTM.

    Returns:
        tuple: Processed (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    # Ensure no missing or infinite values
    if train_data.isnull().values.any() or val_data.isnull().values.any() or test_data.isnull().values.any():
        raise ValueError("Input data contains missing values.")
    if np.isinf(train_data.values).any() or np.isinf(val_data.values).any() or np.isinf(test_data.values).any():
        raise ValueError("Input data contains infinite values.")
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_scaled = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns, index=train_data.index)
    val_data_scaled = pd.DataFrame(scaler.transform(val_data), columns=val_data.columns, index=val_data.index)
    test_data_scaled = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns, index=test_data.index)

    # Split data into sequences
    X_train, y_train = construct_lstm_data(train_data_scaled.to_numpy(), seq_size)
    X_val, y_val = construct_lstm_data(val_data_scaled.to_numpy(), seq_size)
    X_test, y_test = construct_lstm_data(test_data_scaled.to_numpy(), seq_size)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler