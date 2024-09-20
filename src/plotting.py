import pandas as pd
import mplfinance as mpf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
# import logging
import matplotlib.dates as mdates
import numpy as np
# logging.basicConfig(level=logging.DEBUG)

def plot_stats(data: dict[str, pd.DataFrame], stock) -> tuple[plt.Figure | None, go.Figure]:
    try:
        # logging.debug(f"Entering plot_stats function with stock: {stock}")
        
        if not data:
            raise ValueError("Data is empty")

        if not isinstance(data, dict):
            raise ValueError("Data is not a dictionary")

        if not all(isinstance(df, pd.DataFrame) for df in data.values()):
            raise ValueError("Data is not a dictionary of DataFrames")

        if not all('change_percent' in df.columns for df in data.values()):
            raise ValueError("Data does not contain 'change_percent' column")

        if stock not in data:
            raise ValueError(f"Stock '{stock}' not found in data")

        df = data[stock]
        df.index = pd.to_datetime(df.index)

        colors = {stock: mcolors.to_hex(plt.cm.tab10(0))}

        fig = make_subplots(
            rows=2, cols=2, 
            subplot_titles=(
                'Daily Percentage Change',
                'Distribution of Daily Percentage Change',
                'Boxplot and Violin Plot of Daily Percentage Change'
            ),
            specs=[[{"rowspan": 1}, {}], [{"colspan": 2}, None]]
        )
        fig_candlestick = None
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            s = mpf.make_mpf_style(base_mpf_style='yahoo')
            fig_candlestick, ax = mpf.plot(df, type='candle', style=s, 
                    title=f'Candlestick Chart for {stock}',
                    ylabel='Price ($)', 
                    mav=(10,20,50), 
                    volume=True, 
                    returnfig=True,
                    figsize=(10,5),
                    )

        # Daily Percentage Change
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['change_percent'], 
                mode='lines', 
                name=f'{stock} stock', 
                line=dict(color=colors[stock], width=2),
                showlegend=True
            ), 
            row=1, col=1
        )

        # Distribution of Daily Percentage Change
        fig.add_trace(
            go.Histogram(
                x=df['change_percent'], 
                name=f'{stock} stock', 
                opacity=0.7, 
                nbinsx=200, 
                marker_color=colors[stock],
                showlegend=True
            ), 
            row=1, col=2 
        )

        # Boxplot
        fig.add_trace(
            go.Box(
                y=df['change_percent'], 
                name=f'{stock} stock', 
                line=dict(color=colors[stock]),
                opacity=0.6,
                showlegend=False
            ),
            row=2, col=1
        )

        # Violin plot
        fig.add_trace(
            go.Violin(
                y=df['change_percent'], 
                name=f'{stock} stock', 
                line=dict(color=colors[stock]),
                box_visible=True, 
                meanline_visible=True, 
                points='all', 
                opacity=0.6,
                showlegend=False
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title='Volatility',
            template='plotly_white'
        )

        fig.update_xaxes(title_text='Date', row=1, col=1)
        fig.update_yaxes(title_text='Percentage Change', row=1, col=1)

        fig.update_xaxes(title_text='Percentage Change', row=1, col=2)
        fig.update_yaxes(title_text='Frequency', row=1, col=2)

        fig.update_xaxes(title_text='Stock', row=2, col=1)
        fig.update_yaxes(title_text='Percentage Change', row=2, col=1)

        # logging.debug(f"Fig layout title: {fig.layout.title}")
        # logging.debug("Finished creating plots")
        return fig_candlestick, fig
    except Exception as e:
        # logging.error(f"Error in plot_stats: {str(e)}")
        raise ValueError(f"Error in plot_stats: {str(e)}") from e


def plot_splitted_stock_data(dataframes, colors):
    fig, ax = plt.subplots(figsize=(16, 9))
    legend_labels = []
    for label, df in dataframes.items():
        df.plot(ax=ax, label=label, color=colors[label], linewidth=1.5, legend=False)  # Disable automatic legend. Produces errors
        legend_labels.append(label)  # Store the label for the manual legend
    ax.legend(legend_labels,fontsize=12)

    ax.set_title('Evolution of Prices Over Time', fontsize=18)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price (in USD)', fontsize=16)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='--', linewidth=0.5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    return fig

def plot_training_performance(history):
    """
    Plots the training and validation loss using Plotly.

    Args:
    - history: The history object returned by model.fit() containing loss and val_loss.

    Returns:
    - A Plotly figure showing the training and validation loss.
    """

    # Create a Plotly figure
    fig = go.Figure()
    colors = mcolors.to_hex(plt.cm.tab10(0))
    # Add trace for training loss
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history['loss']) + 1)),
        y=history['loss'],
        mode='lines',
        name='Training Loss',
        line=dict(color='blue')
    ))

    # Add trace for validation loss
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history['val_loss']) + 1)),
        y=history['val_loss'],
        mode='lines',
        name='Validation Loss',
        line=dict(color='red')
    ))

    # Set title and labels
    fig.update_layout(
        title="LSTM Model Performance",
        xaxis_title="Epochs",
        yaxis_title="Loss",
        legend_title="Loss Type",
        width=900,  # Adjust the size as needed
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig

# def plot_actual_vs_predicted_stock_data(actual_data, predicted_data, colors):
#     """
#     Plots the actual vs predicted stock data for training, validation, and test sets.

#     Parameters:
#     - actual_data: dict containing 'Train', 'Validation', and 'Test' actual dataframes.
#     - predicted_data: dict containing 'Train', 'Validation', and 'Test' numpy arrays (predictions).
#     - colors: dict defining the colors for each dataset ('Train', 'Validation', 'Test').

#     Returns:
#     - fig: Matplotlib figure object for further customization or saving.
#     """
#     fig, ax = plt.subplots(figsize=(16, 9))
#     legend_labels = []
    
#     # Plot actual data
#     for label, df in actual_data.items():
#         df.plot(ax=ax, label=f"{label} Actual", color=colors[label], linewidth=1.5, legend=False)
#         legend_labels.append(f"{label} Actual")

#     # Plot predicted data
#     for label, predictions in predicted_data.items():
#         # Align predictions with the actual data index (usually np arrays, so we align with actual data's index)
#         if len(predictions) == len(actual_data[label]):
#             ax.plot(actual_data[label].index, predictions, linestyle='--', color=colors[label], linewidth=2, label=f"{label} Predicted")
#             legend_labels.append(f"{label} Predicted")
#         else:
#             ax.plot(actual_data[label].index[-len(predictions):], predictions, linestyle='--', color=colors[label], linewidth=2, label=f"{label} Predicted")
#             legend_labels.append(f"{label} Predicted")

#     # Formatting
#     ax.legend(legend_labels, fontsize=12)
#     ax.set_title('Actual vs Predicted Stock Prices Over Time', fontsize=18)
#     ax.set_xlabel('Date', fontsize=14)
#     ax.set_ylabel('Price (in USD)', fontsize=16)
#     ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
#     ax.minorticks_on()
#     ax.grid(True, which='minor', linestyle='--', linewidth=0.5)

#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#     plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
#     plt.xticks(rotation=45)

#     return fig

def plot_actual_vs_predicted_stock_data(actual_data, predicted_data, colors):
    """
    Plots the actual vs predicted stock data for training, validation, and test sets using Plotly.

    Parameters:
    - actual_data: dict containing 'Train', 'Validation', and 'Test' actual dataframes.
    - predicted_data: dict containing 'Train', 'Validation', and 'Test' numpy arrays (predictions).
    - colors: dict defining the colors for each dataset ('Train', 'Validation', 'Test').

    Returns:
    - fig: Plotly figure object for rendering in Streamlit.
    """
    fig = go.Figure()

    # Plot actual data (thicker and more transparent)
    for label, df in actual_data.items():
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.values.squeeze(),  # assuming df has a single column (for stock prices)
            mode='lines',
            name=f"{label} Actual",
            line=dict(color=colors[label], width=4),  # Thicker line
            opacity=0.6  # More transparency
        ))

    # Plot predicted data (thinner and more contrasting)
    for label, predictions in predicted_data.items():
        if len(predictions) == len(actual_data[label]):
            # Align the predictions to the same index as the actual data
            fig.add_trace(go.Scatter(
                x=actual_data[label].index,
                y=predictions,
                mode='lines',
                name=f"{label} Predicted",
                line=dict(color=colors[label], width=2, dash='dash')  # Thinner and dashed line
            ))
        else:
            # Align to the end of the actual data (in case predictions are shorter)
            fig.add_trace(go.Scatter(
                x=actual_data[label].index[-len(predictions):],
                y=predictions,
                mode='lines',
                name=f"{label} Predicted",
                line=dict(color=colors[label], width=2, dash='dash')  # Thinner and dashed line
            ))

    # Update layout for title, labels, and formatting
    fig.update_layout(
        title="Actual vs Predicted Stock Prices Over Time",
        xaxis_title="Date",
        yaxis_title="Price (in USD)",
        legend_title="Legend",
        xaxis=dict(tickformat='%Y-%m', tickangle=-45),
        template='plotly_white'
    )

    return fig