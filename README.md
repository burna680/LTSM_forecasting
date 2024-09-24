**LSTM Stock Price Forecasting Web Application**
=====================================================
![alt text](misc/stocks.gif)

**Description**
---------------
This project is a web application built using Streamlit that utilizes Long Short-Term Memory (LSTM) networks to forecast stock prices. The app allows users to gather data, analyze and preprocess it, train a model, and view the results.

![alt text](misc/LTSM_gif.gif)

**Table of Contents**
----------------------

* [Installation and Running the Project](#installation-and-running-the-project)
* [Usage Guide](#usage-guide)
* [License](#license)
* [Credits](#credits)

**Installation and Running the Project**
------------------------------------------

To install and run the project, follow these steps:

1. Clone the repository using `git clone https://github.com/burna680/LTSM_forecasting.git`
2. Create a virtual environment using the `create_env.sh` script:
```bash
bash create_env.sh
```
This script will create a virtual environment with the necessary packages and requirements.

**create_env.sh**
```bash
#!/bin/bash

# Create a virtual environment
python3 -m venv LTSM_forecasting

# Activate the virtual environment
source LTSM_forecasting/bin/activate

# Install required packages
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate
```
3. Activate the virtual environment:
```bash
source LTSM_forecasting/bin/activate
```
4. Run the project using:
```bash
streamlit run main.py
```

**Usage Guide**
--------------

1. **Gather Data**: Use the app to gather historical stock price data for a specific stock ticker.
2. **Analyze and Preprocess Data**: The app will analyze and preprocess the data for training the LSTM model.
3. **Train Model**: Train the LSTM model using the preprocessed data.
4. **View Results**: View the forecasted stock prices and compare them to the actual prices.

**Contributing**
--------------

We welcome contributions to this project! If you're interested in contributing, please follow these steps:

1. Fork the repository and create a new branch for your feature or bug fix.
2. Make your changes and commit them with a clear and descriptive commit message.
3. Open a pull request to the main branch, including a brief description of your changes.
4. We'll review your pull request and provide feedback or merge it into the main branch.

**Credits**
-------

* This project was built using Streamlit and utilizes various libraries such as `yfinance`, `pandas`, and `scikit-learn`.
* The LSTM forecasting model was implemented using the `keras` library.
* The app's UI was designed using Streamlit's built-in components and `plotly` for visualization.


**License**
-------

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.