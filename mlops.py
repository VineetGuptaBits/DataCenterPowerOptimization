import re
import mlflow
import mlflow.sklearn # Import for logging scikit-learn models (XGBoost)
import mlflow.keras # Import for logging Keras models (RNN, LSTM, GRU)
from bayes_opt import BayesianOptimization
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, r2_score # For XGBoost evaluation
# from model_train import ModelTrainTestOptimize # This import is not needed in mlops.py itself
from datetime import datetime
import numpy as np # For flattening arrays

import os
import shutil
from datetime import datetime
import tensorflow as tf
import joblib
import subprocess
import socket

def start_mlflow_ui_if_not_running():
    """
    Checks if the MLflow UI is running on the default port (5000)
    and starts it in a non-blocking way if it's not.
    """
    host = '127.0.0.1'
    port = 5000
    try:
        # Try to create a socket connection to the host and port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)  # Set a short timeout
            s.connect((host, port))
            print(f"MLflow UI is already running at http://{host}:{port}")
    except (ConnectionRefusedError, socket.timeout):
        # If the connection is refused or times out, the port is likely free
        print(f"MLflow UI not detected on http://{host}:{port}. Attempting to start...")
        try:
            # Use subprocess.Popen to start the UI in the background
            # This is a non-blocking call, so the script can continue
            #subprocess.Popen(["mlflow", "ui"])
            subprocess.Popen(["C:\\Users\\guptav31\\AppData\\Roaming\\Python\\Python310\\Scripts\\mlflow.exe", "ui"])
            print("MLflow UI started in the background.")
            print(f"You can now access it at http://{host}:{port}")
        except FileNotFoundError:
            # Handle the case where 'mlflow' command is not in the system's PATH
            print("Error: 'mlflow' command not found. Please ensure MLflow is installed and in your system's PATH.")
        except Exception as e:
            print(f"An unexpected error occurred while starting MLflow UI: {e}")
            
class MlflowModel():
    def __init__(self,url,exp):
        self.url = url
        self.formatted_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M")

        self.exp_name = "Datacenter_exp_"+exp+"_"+self.formatted_datetime
        self.arti_repo = "./mlflow-run/"+exp+"/"+self.formatted_datetime

    def create_exp(self):
        mlflow.set_tracking_uri(self.url)
        client = MlflowClient()

        try:
            self.exp_id = client.create_experiment(self.exp_name,artifact_location=self.arti_repo)
        except:
            self.exp_id = client.get_experiment_by_name(self.exp_name).experiment_id
        
        return client, self.exp_id

    def start_experiment(self,run_name):
        with mlflow.start_run(experiment_id=self.exp_id,run_name = run_name) as exp_run:
            exp_run_id = exp_run.info.run_id
        return exp_run_id

    def exp_hyperparamter_tuning(self,mdl_nm,objective_func,pbounds,run_id):

        with mlflow.start_run(experiment_id=self.exp_id,run_id=run_id):
            with mlflow.start_run(experiment_id=self.exp_id,nested=True, run_name=f"BayesianOptimization_{mdl_nm}") as nested_run:
                optimizer = BayesianOptimization(f=objective_func,pbounds=pbounds,random_state=1)
                optimizer.maximize(init_points=5,n_iter=2) # Reduced n_iter for faster simulation
                
                best_params = optimizer.max['params'].copy()
                # Log parameters, handling potential non-string values
                for key, value in best_params.items():
                    if isinstance(value, (float, int)):
                        mlflow.log_param(key, value)
                    else:
                        mlflow.log_param(key, str(value)) # Convert other types to string for logging

                # Add common parameters for Keras models if applicable, or specific for XGBoost
                if mdl_nm in ['rnn', 'lstm', 'gru']:
                    mlflow.log_param('optimizer_type', 'bayesian_optimization')
                    mlflow.log_param('loss_function', 'mse')
                    mlflow.log_param('metrics_evaluated', ['mse'])
                    mlflow.log_param('epochs', '100')
                    mlflow.log_param('batch_size', '32')
                elif mdl_nm == 'xgboost':
                    mlflow.log_param('objective_function', 'reg:squarederror')
                    mlflow.log_param('metrics_evaluated', ['rmse']) # XGBoost often uses RMSE
                
                mlflow.log_metric("best_mse", -optimizer.max['target']) # Log positive MSE
                
        return -optimizer.max['target'] , optimizer # Return positive MSE

    def exp_model_selection(self, model, mdl_nm, exp_id, optimizer, X_train_final, Y_train_final, X_test_final, Y_test_final):
        mdl_nm = mdl_nm.lower()
        with mlflow.start_run(experiment_id=exp_id, run_name=f"Final_Model_{mdl_nm}") as run:
            mlflow.log_params(optimizer.max['params'])

            if mdl_nm in ['rnn', 'lstm', 'gru']:
                # For Keras models, fit here if not already fitted to full data
                # The model is already fitted in ModelSelection.final_model_selection before being passed here,
                # so this fit call might be redundant if the model is already trained.
                # If you want to re-train the final model on full training data, keep this.
                # Otherwise, remove it. For now, keeping it as per original code structure.
                model.fit(X_train_final, Y_train_final, epochs=100, batch_size=32, validation_data=(X_test_final, Y_test_final), verbose=1)
                mlflow.keras.log_model(model, f"Datacenter_{mdl_nm}")
                
                # Evaluate Keras model
                loss, mse = model.evaluate(X_test_final, Y_test_final, verbose=0)
                y_pred = model.predict(X_test_final)
                r2 = r2_score(Y_test_final.flatten(), y_pred.flatten())

            elif mdl_nm == 'xgboost':
                # XGBoost model is already fitted in ModelSelection.final_model_selection,
                # so no need to fit again here.
                mlflow.sklearn.log_model(model, f"Datacenter_{mdl_nm}")

                # Evaluate for XGBoost
                # Ensure X_test_final is flattened for XGBoost prediction
                if X_test_final.ndim == 3:
                    X_test_flat = X_test_final.reshape(X_test_final.shape[0], -1)
                else:
                    X_test_flat = X_test_final

                y_pred = model.predict(X_test_flat)
                mse = mean_squared_error(Y_test_final.flatten(), y_pred.flatten())
                r2 = r2_score(Y_test_final.flatten(), y_pred.flatten())
                loss = mse # Use MSE as "loss" for consistency in logging

            else:
                raise ValueError(f"Invalid model name provided: {mdl_nm}. Choose 'rnn', 'lstm', 'gru', or 'xgboost'.")

            mlflow.log_metric("test_loss", loss)
            mlflow.log_metric("test_mse", mse)
            mlflow.log_metric("test_r2_score", r2) # Log R2 score
            print(f"Run ID: {run.info.run_uuid}")
        
        mlflow.end_run()

        return run.info.run_uuid

    def upload_model(self,model_name,run_id):
        # The model_name passed here is like 'RNN', 'LSTM', 'GRU', 'XGBoost'.
        # The artifact path will be 'Datacenter_RNN', 'Datacenter_LSTM', etc.
        artifact_path = f"Datacenter_{model_name}" 
        model_uri = f"runs:/{run_id}/{artifact_path}"
        print(f"Attempting to register model from URI: {model_uri}")
        
        ret = 1
        version = None
        m_name = None
        try:
            result = mlflow.register_model(model_uri, model_name)
            if result:
                ret = 0
                version = result.version
                m_name = result.name
            else:
                ret = 1
        except Exception as e:
            print(f"Error registering model: {e}")
            ret = 1

        return ret , version , m_name

def get_model(version, model_name):
    client = MlflowClient()
    loaded_model = None
    try:
        # Get model version details from MLflow Registry
        #model_version_info = client.get_model_version(name=model_name, version=version)
        #source_uri = model_version_info.source
        #source_uri = "file:///C:/Users/guptav31/mlflow-run/new/31_07_2025_14_33/1ca3b0da62d24aaa885e3442cdc78cc7/artifacts/Datacenter_gru/data"
        source_uri = "file:///C:/Users/guptav31/mlflow-run/new/31_07_2025_17_18/1dfca82d175746e2b5aa15b5cce41360/artifacts/Datacenter_lstm/data/model.keras"
        print(f"Source URI: {source_uri}")
        loaded_model = mlflow.keras.load_model(source_uri)
        # Determine model flavor and load accordingly
        # A more robust way would be to query MLflow for the flavor, but for now,
        # we'll infer based on model_name string.
        #if "RNN" in model_name.upper() or "LSTM" in model_name.upper() or "GRU" in model_name.upper():
        #    loaded_model = mlflow.keras.load_model(source_uri)
        #    print(f"Keras model '{model_name}' loaded successfully.")
        #elif "XGBOOST" in model_name.upper():
        #    loaded_model = mlflow.sklearn.load_model(source_uri)
        #    print(f"Scikit-learn (XGBoost) model '{model_name}' loaded successfully.")
        #else:
        #    print(f"Warning: Could not determine specific model type for '{model_name}'. Attempting generic pyfunc load.")
        #    loaded_model = mlflow.pyfunc.load_model(source_uri) # Generic load
        #    print(f"Generic pyfunc model '{model_name}' loaded successfully.")

    except mlflow.exceptions.MlflowException as e:
        print(f"MLflow Error loading model: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
    
    return loaded_model


# --- Local Model Storage Configuration ---
MODEL_BASE_DIR = "models"


def save_model(model, model_name, model_type='keras'):
    """
    Saves a machine learning model locally with a versioned filename and a 'latest' link.

    Args:
        model: The model object to save (e.g., tf.keras.Model, xgboost.Booster).
        model_name (str): The base name for the model (e.g., 'LSTM', 'XGBoost').
        model_type (str): The type of model. Must be 'keras' or 'xgboost'.
    """
    model_dir = os.path.join(MODEL_BASE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_type == 'keras':
        file_extension = '.keras'
        save_function = lambda path: model.save(path)
    elif model_type == 'xgboost':
        file_extension = '.joblib'
        save_function = lambda path: joblib.dump(model, path)
    else:
        print(f"Error: Unsupported model type '{model_type}'. Model not saved.")
        return None, None
        
    versioned_filename = f"{model_name}_{timestamp}{file_extension}"
    versioned_path = os.path.join(model_dir, versioned_filename)
    
    # Save the model
    try:
        save_function(versioned_path)
        print(f"Model saved successfully to: {versioned_path}")
    except Exception as e:
        print(f"Error saving model to '{versioned_path}': {e}")
        return None, None
    
    # Create or update a link to the latest version
    latest_filename = f"latest{file_extension}"
    latest_path = os.path.join(model_dir, latest_filename)
    try:
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.link(versioned_path, latest_path)
        print(f"Updated 'latest' link to: {latest_path}")
    except (OSError, NotImplementedError):
        # Fallback to file copy if symlinks are not supported
        shutil.copyfile(versioned_path, latest_path)
        print(f"Updated 'latest' model by copying to: {latest_path}")

    return versioned_path, latest_path


def load_latest_model(model_name, model_type='keras'):
    """
    Loads the latest version of a model from a local file.

    Args:
        model_name (str): The base name for the model (e.g., 'LSTM', 'XGBoost').
        model_type (str): The type of model. Must be 'keras' or 'xgboost'.

    Returns:
        The loaded model object, or None if not found.
    """
    if model_type == 'keras':
        file_extension = '.keras'
        load_function = lambda path: tf.keras.models.load_model(path)
    elif model_type == 'xgboost':
        file_extension = '.joblib'
        load_function = lambda path: joblib.load(path)
    else:
        print(f"Error: Unsupported model type '{model_type}'. Model not loaded.")
        return None

    model_path = os.path.join(MODEL_BASE_DIR, model_name, f"latest{file_extension}")
    
    if not os.path.exists(model_path):
        print(f"Error: No 'latest' model found for '{model_name}' at '{model_path}'.")
        return None
        
    try:
        loaded_model = load_function(model_path)
        print(f"Model '{model_name}' loaded successfully from: {model_path}")
        return loaded_model
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")

        return None

