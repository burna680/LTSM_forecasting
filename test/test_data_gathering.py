import pandas as pd
from src.data_handling import gather_data 
import pytest

def test_default_stocks_and_period():
    data = gather_data()
    assert isinstance(data, dict)
    assert len(data) == 5  # default stocks
    for stock, df in data.items():
        assert isinstance(df, pd.DataFrame)
        assert 'change_percent' in df.columns

def test_custom_stocks_and_period():
    stocks = ['MSFT', 'GOOG']
    period = '2y'
    data = gather_data(stocks, period)
    assert isinstance(data, dict)
    assert len(data) == 2  # custom stocks
    for stock, df in data.items():
        assert isinstance(df, pd.DataFrame)
        assert 'change_percent' in df.columns

def test_invalid_stock_symbol():
    stocks = ['INVALID_STOCK']
    with pytest.raises(ValueError):
        gather_data(stocks)

def test_invalid_period():
    period = ' invalid_period'
    with pytest.raises(ValueError):
        gather_data(period=period)

def test_empty_stock_list():
    stocks = []
    data = gather_data(stocks)
    assert isinstance(data, dict)
    assert len(data) == 0

def test_none_stock_list():
    stocks = None
    data = gather_data(stocks)
    assert isinstance(data, dict)
    assert len(data) == 1  # default stocks

def test_none_period():
    period = None
    data = gather_data(period=period)
    assert isinstance(data, dict)
    assert len(data) == 5  # default stocks

def test_single_stock():
    stock = ['AAPL']
    data = gather_data(stock)
    assert isinstance(data, dict)
    assert len(data) == 1
    assert 'AAPL' in data
    assert isinstance(data['AAPL'], pd.DataFrame)
    assert 'change_percent' in data['AAPL'].columns

def test_multiple_stocks():
    stocks = ['AAPL', 'GOOG', 'MSFT']
    data = gather_data(stocks)
    assert isinstance(data, dict)
    assert len(data) == 3
    for stock in stocks:
        assert stock in data
        assert isinstance(data[stock], pd.DataFrame)
        assert 'change_percent' in data[stock].columns

def test_stock_with_spaces():
    stock = ['AAPL Inc']
    with pytest.raises(ValueError):
        gather_data(stock)

def test_stock_with_special_chars():
    stock = ['AAPL!']
    with pytest.raises(ValueError):
        gather_data(stock)

def test_period_with_spaces():
    period = ' 1y '
    data = gather_data(period=period)
    assert isinstance(data, dict)
    assert len(data) == 5  # default stocks
    for stock, df in data.items():
        assert isinstance(df, pd.DataFrame)
        assert 'change_percent' in df.columns

def test_period_with_special_chars():
    period = '1y!'
    with pytest.raises(ValueError):
        gather_data(period=period)

def test_empty_period():
    period = ''
    with pytest.raises(ValueError):
        gather_data(period=period)

def test_none_stock_and_period():
    stock = None
    period = None
    data = gather_data(stock, period)
    assert isinstance(data, dict)
    assert len(data) == 1  # default stocks
    for stock, df in data.items():
        assert isinstance(df, pd.DataFrame)
        assert 'change_percent' in df.columns

def test_invalid_stock_type():
    stock = 123
    with pytest.raises(TypeError):
        gather_data(stock)

def test_invalid_period_type():
    period = 123
    with pytest.raises(TypeError):
        gather_data(period=period)
