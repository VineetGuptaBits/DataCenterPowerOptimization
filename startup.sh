#!/bin/bash

# Create the directory for MLflow to store data if it doesn't exist.
# This directory is the mount point for the Azure File Share.
mkdir -p /home/mlflow_data/mlruns

# Start the MLflow server in the background, listening on the mounted directory.
# We use 'nohup' and '&' to run it in the background.
nohup mlflow server \
--host 0.0.0.0 \
--port 5000 \
--backend-store-uri file:///home/mlflow_data/mlruns \
--default-artifact-root file:///home/mlflow_data/mlruns &

# Set the MLFLOW_TRACKING_URI environment variable for the Streamlit app.
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Start the Streamlit app, which will run in the foreground.
# Azure App Service will use this process to monitor the health of the app.
python -m streamlit run HomePage.py --server.port 8000 --server.address 0.0.0.0
