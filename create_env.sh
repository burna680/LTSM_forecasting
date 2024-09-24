#!/bin/bash

# Create a virtual environment
python3 -m venv LTSM_forecasting

# Activate the virtual environment
source LTSM_forecasting/bin/activate

# Install required packages
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate