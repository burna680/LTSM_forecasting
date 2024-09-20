import pytest
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import src.plotting as plotting
import logging

logging.basicConfig(level=logging.DEBUG)

def test_plot_stats_empty_data():
    with pytest.raises(ValueError):
        plotting.plot_stats({}, 'stock')

def test_plot_stats_non_dict_data():
    with pytest.raises(ValueError):
        plotting.plot_stats('data', 'stock')

def test_plot_stats_non_dataframe_data():
    data = {'stock': pd.Series([1, 2, 3])}
    with pytest.raises(ValueError):
        plotting.plot_stats(data, 'stock')

def test_plot_stats_missing_change_percent():
    data = {'stock': pd.DataFrame({'col1': [1, 2, 3]})}
    with pytest.raises(ValueError):
        plotting.plot_stats(data, 'stock')

def test_plot_stats_valid_data():
    data = {'stock': pd.DataFrame({
        'change_percent': [0.1, 0.2, 0.3],
        'Open': [10, 20, 30],
        'High': [15, 25, 35],
        'Low': [5, 15, 25],
        'Close': [12, 22, 32],
        'Volume': [1000, 2000, 3000]
    }, index=pd.date_range(start='2024-01-01', periods=3))}
    
    fig_candlestick, fig = plotting.plot_stats(data, 'stock')
    
    logging.debug(f"Fig candlestick type: {type(fig_candlestick)}")
    logging.debug(f"Fig type: {type(fig)}")
    logging.debug(f"Fig layout: {fig.layout}")
    logging.debug(f"Fig layout title: {fig.layout.title}")
    
    assert isinstance(fig_candlestick, plt.Figure)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == 'Volatility'
    
    # Check subplot titles
    assert len(fig.layout.annotations) == 3
    assert fig.layout.annotations[0].text == 'Daily Percentage Change'
    assert fig.layout.annotations[1].text == 'Distribution of Daily Percentage Change'
    assert fig.layout.annotations[2].text == 'Boxplot and Violin Plot of Daily Percentage Change'

    # assert fig.layout.annotations[0].text == 'Daily Percentage Change'
    # assert fig.layout.annotations[1].text == 'Distribution of Daily Percentage Change'
    # assert fig.layout.annotations[2].text == 'Boxplot and Violin Plot of Daily Percentage Change'    # Check if the figure contains correct number of traces
    assert len(fig.data) == 4  # 1 scatter, 1 histogram, 1 box, 1 violin

    # Check if the traces are of the expected types
    assert fig.data[0].type == 'scatter'
    assert fig.data[1].type == 'histogram'
    assert fig.data[2].type == 'box'
    assert fig.data[3].type == 'violin'
    
    # Validate layout titles
    assert fig.layout.xaxis.title.text == 'Date'
    assert fig.layout.yaxis.title.text == 'Percentage Change'

def test_plot_stats_multiple_stocks():
    data = {
        'stock1': pd.DataFrame({
            'change_percent': [0.1, 0.2, 0.3],
            'Open': [10, 20, 30],
            'High': [15, 25, 35],
            'Low': [5, 15, 25],
            'Close': [12, 22, 32],
            'Volume': [1000, 2000, 3000]
        }, index=pd.date_range(start='2024-01-01', periods=3)),
        'stock2': pd.DataFrame({
            'change_percent': [0.4, 0.5, 0.6],
            'Open': [40, 50, 60],
            'High': [45, 55, 65],
            'Low': [35, 45, 55],
            'Close': [42, 52, 62],
            'Volume': [4000, 5000, 6000]
        }, index=pd.date_range(start='2024-01-01', periods=3))
    }
    
    fig_candlestick, fig = plotting.plot_stats(data, 'stock1')
    
    logging.debug(f"Fig candlestick type: {type(fig_candlestick)}")
    logging.debug(f"Fig type: {type(fig)}")
    logging.debug(f"Fig layout: {fig.layout}")
    logging.debug(f"Fig layout title: {fig.layout.title}")
    
    assert isinstance(fig_candlestick, plt.Figure)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == 'Volatility'    
    # Check subplot titles
    assert len(fig.layout.annotations) == 3
    assert fig.layout.annotations[0].text == 'Daily Percentage Change'
    assert fig.layout.annotations[1].text == 'Distribution of Daily Percentage Change'
    assert fig.layout.annotations[2].text == 'Boxplot and Violin Plot of Daily Percentage Change'
    # assert fig.layout.annotations[0].text == 'Daily Percentage Change'
    # assert fig.layout.annotations[1].text == 'Distribution of Daily Percentage Change'
    # assert fig.layout.annotations[3].text == 'Boxplot and Violin Plot of Daily Percentage Change'

    # Check if the figure contains correct number of traces
    assert len(fig.data) == 4  # 1 scatter, 1 histogram, 1 box, 1 violin

    # Check if the traces are of the expected types
    assert fig.data[0].type == 'scatter'
    assert fig.data[1].type == 'histogram'
    assert fig.data[2].type == 'box'
    assert fig.data[3].type == 'violin'
    
    # Validate layout titles
    assert fig.layout.xaxis.title.text == 'Date'
    assert fig.layout.yaxis.title.text == 'Percentage Change'

def test_plot_stats_invalid_stock():
    data = {
        'stock1': pd.DataFrame({'change_percent': [0.1, 0.2, 0.3]}),
        'stock2': pd.DataFrame({'change_percent': [0.4, 0.5, 0.6]})
    }
    with pytest.raises(ValueError):
        plotting.plot_stats(data, 'stock3')

def test_plot_stats_empty_stock():
    data = {
        'stock1': pd.DataFrame({'change_percent': [0.1, 0.2, 0.3]}),
        'stock2': pd.DataFrame({'change_percent': [0.4, 0.5, 0.6]})
    }
    with pytest.raises(ValueError):
        plotting.plot_stats(data, '')

def test_plot_stats_none_stock():
    data = {
        'stock1': pd.DataFrame({'change_percent': [0.1, 0.2, 0.3]}),
        'stock2': pd.DataFrame({'change_percent': [0.4, 0.5, 0.6]})
    }
    with pytest.raises(ValueError):
        plotting.plot_stats(data, None)

def test_plot_stats_invalid_data_type():
    data = 'invalid data'
    with pytest.raises(ValueError):
        plotting.plot_stats(data, 'stock')

def test_plot_stats_invalid_data_structure():
    data = {'stock': 'invalid data'}
    with pytest.raises(ValueError):
        plotting.plot_stats(data, 'stock')