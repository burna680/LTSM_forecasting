**LSTM Forecasting App**
==========================

**Description**
------------------------

This project is a web application built using Streamlit that utilizes Long Short-Term Memory (LSTM) networks to forecast stock prices. The app allows users to gather data, analyze and preprocess it, train a model, and view the results.

**Table of Contents**
----------------------

* [Installation and Running the Project](#installation-and-running-the-project)
* [How to Use the Project](#how-to-use-the-project)
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
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required packages
pip install streamlit
pip install streamlit-option-menu
pip install pandas
pip install yfinance
pip install numpy
pip install scikit-learn
pip install plotly

# Deactivate the virtual environment
deactivate
```
3. Activate the virtual environment:
```bash
source venv/bin/activate
```
4. Run the project using:
```bash
streamlit run main.py
```
**How to Use the Project**
---------------------------

1. Open the app in your web browser by navigating to `http://localhost:8501`
2. Select the first page from the menu to perform the following actions:
	* Gather data: Use the `Data gathering` page to retrieve stock data using the `yfinance` library.
	* Analyze and preprocess data: Use the `Data analysis` and `Data preprocessing` pages to explore and transform the data.
	* Train a model: Use the `Model training` page to train an LSTM model on the preprocessed data.
	* View results: Use the `Results` page to view the forecasted stock prices.

**Credits**
------------

* This project was built using Streamlit and utilizes various libraries such as `yfinance`, `pandas`, and `scikit-learn`.
* The LSTM forecasting model was implemented using the `keras` library.
* The app's UI was designed using Streamlit's built-in components and `plotly` for visualization.


## Steps in every ML project

1. Data gathering
2. Data analysis
3. Data transformation/preparation
4. Model training & development 
5. Model validation 
6. Model serving 
7. Model monitoring 
8. Model re-training 

**Important aspects when deploying a scalable pipeline**
-----------
- Configuration
- Automation
- Data verification
- Testing and debugging
- Resource management
- Process and metadata management
- Serving infrastructure
