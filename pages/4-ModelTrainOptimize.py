import streamlit as st
from model_train import ModelTrainTestOptimize, ModelSelection
from mlops import MlflowModel, save_model, start_mlflow_ui_if_not_running
from matplotlib import pyplot as plt
import numpy as np
import os
import joblib
import pandas as pd # Added pandas for DataFrame display

# --- Configuration ---
start_mlflow_ui_if_not_running()
MLFLOW_URL = "http://127.0.0.1:5000/"
LOOK_BACK_WINDOW = 10 # Not used directly in the provided snippet but kept for context

PBOUNDS = { # For RNN, LSTM, GRU
    'num_layers': (1, 5),
    'units': (32, 128),
    'dropout_rate': (0.1, 0.5),
    'learning_rate': (0.0001, 0.01)
}

XGBOOST_PBOUNDS = { # Specific for XGBoost
    'n_estimators': (50, 200),
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0)
}

# --- Helper function for plotting ---
def model_plotting(y_test, y_pred, title):
    """
    Plots the actual vs. predicted values.
    Assumes y_test and y_pred are already inverse-transformed to the original scale.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test, label='Actual (Inversed)')
    ax.plot(y_pred, label='Predicted (Inversed)')
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Power consumption Usage (Original Scale)')
    ax.grid(True)
    return fig

# --- Streamlit Application ---
st.set_page_config(page_title="Time Series Model Training & Evaluation", layout="wide" ,page_icon="🧠")

st.title("🧠 Time Series Model Training and Evaluation")

st.write("""
This application trains and evaluates RNN, LSTM, GRU, and **XGBoost** models for time series prediction
using hyperparameter tuning with Bayesian Optimization and MLflow for experiment tracking.
""")

# --- Initialize session state variables if they don't exist ---
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'training_complete' not in st.session_state:
    st.session_state['training_complete'] = False
if 'train_X' not in st.session_state:
    st.session_state['train_X'] = None
if 'train_y' not in st.session_state:
    st.session_state['train_y'] = None
if 'test_X' not in st.session_state:
    st.session_state['test_X'] = None
if 'test_y' not in st.session_state:
    st.session_state['test_y'] = None

# RNN/LSTM/GRU specific
if 'rnn_y_pred' not in st.session_state:
    st.session_state['rnn_y_pred'] = None
if 'rnn_y_test' not in st.session_state:
    st.session_state['rnn_y_test'] = None
if 'lstm_y_pred' not in st.session_state:
    st.session_state['lstm_y_pred'] = None
if 'lstm_y_test' not in st.session_state:
    st.session_state['lstm_y_test'] = None
if 'gru_y_pred' not in st.session_state:
    st.session_state['gru_y_pred'] = None
if 'gru_y_test' not in st.session_state:
    st.session_state['gru_y_test'] = None
if 'rnn_best_mse' not in st.session_state:
    st.session_state['rnn_best_mse'] = float('inf')
if 'lstm_best_mse' not in st.session_state:
    st.session_state['lstm_best_mse'] = float('inf')
if 'gru_best_mse' not in st.session_state:
    st.session_state['gru_best_mse'] = float('inf')
if 'rnn_optimizer' not in st.session_state:
    st.session_state['rnn_optimizer'] = None
if 'lstm_optimizer' not in st.session_state:
    st.session_state['lstm_optimizer'] = None
if 'gru_optimizer' not in st.session_state:
    st.session_state['gru_optimizer'] = None

# XGBoost specific
if 'xgb_y_pred' not in st.session_state:
    st.session_state['xgb_y_pred'] = None
if 'xgb_y_test' not in st.session_state:
    st.session_state['xgb_y_test'] = None
if 'xgb_best_mse' not in st.session_state:
    st.session_state['xgb_best_mse'] = float('inf')
if 'xgb_optimizer' not in st.session_state:
    st.session_state['xgb_optimizer'] = None

# General
if 'obj_ModelTrainTestOptimize' not in st.session_state:
    st.session_state['obj_ModelTrainTestOptimize'] = None
if 'scaler_X' not in st.session_state:
    st.session_state['scaler_X'] = None
if 'scaler_y' not in st.session_state:
    st.session_state['scaler_y'] = None
if 'final_y_pred_inversed' not in st.session_state:
    st.session_state['final_y_pred_inversed'] = None
if 'final_y_test_inversed' not in st.session_state:
    st.session_state['final_y_test_inversed'] = None


# --- Data Loading Section ---
st.header("1. Load Data")

uploaded_train_X_path = 'preprocessed_data/train_X.npy'
uploaded_train_y_path = 'preprocessed_data/train_y.npy'
uploaded_test_X_path = 'preprocessed_data/test_X.npy'
uploaded_test_y_path = 'preprocessed_data/test_y.npy'

scaler_dir = "scaler_data"
scaler_X_path = os.path.join(scaler_dir, "scaler_X.joblib")
scaler_y_path = os.path.join(scaler_dir, "scaler_y.joblib")


if not st.session_state['data_loaded']:
    if st.button("Load Preprocessed Data"):
        try:
            train_X = np.load(uploaded_train_X_path, allow_pickle=True)
            train_y = np.load(uploaded_train_y_path, allow_pickle=True)
            test_X = np.load(uploaded_test_X_path, allow_pickle=True)
            test_y = np.load(uploaded_test_y_path, allow_pickle=True)

            st.session_state['data_loaded'] = True
            st.session_state['train_X'] = train_X
            st.session_state['train_y'] = train_y
            st.session_state['test_X'] = test_X
            st.session_state['test_y'] = test_y

            if os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
                try:
                    loaded_scaler_X = joblib.load(scaler_X_path)
                    loaded_scaler_y = joblib.load(scaler_y_path)
                    st.session_state['scaler_X'] = loaded_scaler_X
                    st.session_state['scaler_y'] = loaded_scaler_y
                    st.success(f"MinMaxScalers loaded successfully from '{scaler_X_path}' and '{scaler_y_path}'!")
                except Exception as e:
                    st.error(f"Error loading scalers: {e}")
            else:
                st.warning(f"One or both scalers not found at '{scaler_X_path}' or '{scaler_y_path}'. Please go to the Data Preprocessing page and save the scalers first.")

            st.success("Data loaded and scalers (if found) retrieved successfully!")
            st.write(f"Train X shape: {train_X.shape}")
            st.write(f"Train y shape: {train_y.shape}")
            st.write(f"Test X shape: {test_X.shape}")
            st.write(f"Test y shape: {test_y.shape}")

        except FileNotFoundError:
            st.error(f"Error: Make sure the data files are in the '{os.path.dirname(uploaded_train_X_path)}/' directory.")
        except Exception as e:
            st.error(f"An error occurred while loading data: {e}")
else:
    st.success("Data already loaded.")
    if st.session_state['train_X'] is not None:
        st.write(f"Train X shape: {st.session_state['train_X'].shape}")
        st.write(f"Train y shape: {st.session_state['train_y'].shape}")
        st.write(f"Test X shape: {st.session_state['test_X'].shape}")
        st.write(f"Test y shape: {st.session_state['test_y'].shape}")


# --- Model Training Section ---
st.header("2. Train and Evaluate Models")

if st.session_state['data_loaded']:
    if not st.session_state['training_complete']:
        if st.button("Start Model Training and Hyperparameter Tuning"):
            st.info("Training initiated... This may take a while depending on your dataset size and hardware.")

            train_X = st.session_state['train_X']
            train_y = st.session_state['train_y']
            test_X = st.session_state['test_X']
            test_y = st.session_state['test_y']
            
            scaler_y_obj = st.session_state['scaler_y'] 

            if scaler_y_obj is None:
                st.error("Error: The scaler for the target variable (y) is not available. Please ensure data preprocessing is done and `scaler_y.joblib` is saved and loaded correctly.")
            else:
                obj_ModelTrainTestOptimize = ModelTrainTestOptimize(train_X, train_y, test_X, test_y)

                mlflow_obj = MlflowModel(MLFLOW_URL, 'new')
                client, exp_id = mlflow_obj.create_exp()
                hyper_run = mlflow_obj.start_experiment('HyperParameterTuning')

                st.subheader("Hyperparameter Tuning Results (Best MSE)")

                # RNN
                with st.spinner("Tuning RNN model..."):
                    rnn_best_mse, rnn_optimizer = mlflow_obj.exp_hyperparamter_tuning('rnn', obj_ModelTrainTestOptimize.rnn_objective, PBOUNDS, hyper_run)
                    st.success(f"RNN Best MSE: {rnn_best_mse:.4f}")

                # LSTM
                with st.spinner("Tuning LSTM model..."):
                    lstm_best_mse, lstm_optimizer = mlflow_obj.exp_hyperparamter_tuning('lstm', obj_ModelTrainTestOptimize.lstm_objective, PBOUNDS, hyper_run)
                    st.success(f"LSTM Best MSE: {lstm_best_mse:.4f}")

                # GRU
                with st.spinner("Tuning GRU model..."):
                    gru_best_mse, gru_optimizer = mlflow_obj.exp_hyperparamter_tuning('gru', obj_ModelTrainTestOptimize.gru_objective, PBOUNDS, hyper_run)
                    st.success(f"GRU Best MSE: {gru_best_mse:.4f}")

                # XGBoost - NEW
                with st.spinner("Tuning XGBoost model..."):
                    xgb_best_mse, xgb_optimizer = mlflow_obj.exp_hyperparamter_tuning('xgboost', obj_ModelTrainTestOptimize.xgboost_objective, XGBOOST_PBOUNDS, hyper_run)
                    st.success(f"XGBoost Best MSE: {xgb_best_mse:.4f}")


                st.subheader("Model Predictions")

                st.info("Generating predictions for trained models and inverse transforming...")
                
                rnn_y_pred, rnn_y_test = obj_ModelTrainTestOptimize.model_fitting_testing(rnn_optimizer, 'rnn', scaler_y_obj)
                lstm_y_pred, lstm_y_test = obj_ModelTrainTestOptimize.model_fitting_testing(lstm_optimizer, 'lstm', scaler_y_obj)
                gru_y_pred, gru_y_test = obj_ModelTrainTestOptimize.model_fitting_testing(gru_optimizer, 'gru', scaler_y_obj)
                xgb_y_pred, xgb_y_test = obj_ModelTrainTestOptimize.model_fitting_testing(xgb_optimizer, 'xgboost', scaler_y_obj) # NEW

                st.session_state['rnn_y_pred'] = rnn_y_pred
                st.session_state['rnn_y_test'] = rnn_y_test
                st.session_state['lstm_y_pred'] = lstm_y_pred
                st.session_state['lstm_y_test'] = lstm_y_test
                st.session_state['gru_y_pred'] = gru_y_pred
                st.session_state['gru_y_test'] = gru_y_test
                st.session_state['xgb_y_pred'] = xgb_y_pred # NEW
                st.session_state['xgb_y_test'] = xgb_y_test # NEW

                st.session_state['rnn_best_mse'] = rnn_best_mse
                st.session_state['lstm_best_mse'] = lstm_best_mse
                st.session_state['gru_best_mse'] = gru_best_mse
                st.session_state['xgb_best_mse'] = xgb_best_mse # NEW

                st.session_state['obj_ModelTrainTestOptimize'] = obj_ModelTrainTestOptimize
                st.session_state['rnn_optimizer'] = rnn_optimizer
                st.session_state['lstm_optimizer'] = lstm_optimizer
                st.session_state['gru_optimizer'] = gru_optimizer
                st.session_state['xgb_optimizer'] = xgb_optimizer # NEW

                st.session_state['training_complete'] = True

                st.success("Model training and evaluation complete!")
    else:
        st.info("Model training and hyperparameter tuning already completed. See results below.")
else:
    st.warning("Please load the preprocessed data first to enable model training.")


# --- Visualization and Best Model Selection ---
st.header("3. Visualize Results and Select Best Model")

if st.session_state['training_complete']:
    rnn_y_pred = st.session_state['rnn_y_pred']
    rnn_y_test = st.session_state['rnn_y_test']
    lstm_y_pred = st.session_state['lstm_y_pred']
    lstm_y_test = st.session_state['lstm_y_test']
    gru_y_pred = st.session_state['gru_y_pred']
    gru_y_test = st.session_state['gru_y_test']
    xgb_y_pred = st.session_state['xgb_y_pred'] # NEW
    xgb_y_test = st.session_state['xgb_y_test'] # NEW

    rnn_best_mse = st.session_state['rnn_best_mse']
    lstm_best_mse = st.session_state['lstm_best_mse']
    gru_best_mse = st.session_state['gru_best_mse']
    xgb_best_mse = st.session_state['xgb_best_mse'] # NEW

    obj_ModelTrainTestOptimize = st.session_state['obj_ModelTrainTestOptimize']
    rnn_optimizer = st.session_state['rnn_optimizer']
    lstm_optimizer = st.session_state['lstm_optimizer']
    gru_optimizer = st.session_state['gru_optimizer']
    xgb_optimizer = st.session_state['xgb_optimizer'] # NEW

    st.subheader("Prediction Plots (Inversed Values)")
    st.write("The plots below display the actual and predicted power consumption, which have been inverse-transformed back to their original scale.")

    col1, col2, col3, col4 = st.columns(4) # Added a column for XGBoost

    with col1:
        st.pyplot(model_plotting(rnn_y_test, rnn_y_pred, "RNN Test prediction: Actual vs. Predicted"))
    with col2:
        st.pyplot(model_plotting(lstm_y_test, lstm_y_pred, "LSTM Test prediction: Actual vs. Predicted"))
    with col3:
        st.pyplot(model_plotting(gru_y_test, gru_y_pred, "GRU Test prediction: Actual vs. Predicted"))
    with col4: # NEW
        st.pyplot(model_plotting(xgb_y_test, xgb_y_pred, "XGBoost Test prediction: Actual vs. Predicted"))


    st.subheader("Best Model Selection and Saving Model for Inference")
    model_selection_obj = ModelSelection(obj_ModelTrainTestOptimize)

    # Determine the best model based on lowest MSE (lower is better)
    # Using a dictionary to easily find the min MSE and corresponding model
    best_mses = {
        'RNN': rnn_best_mse,
        'LSTM': lstm_best_mse,
        'GRU': gru_best_mse,
        'XGBoost': xgb_best_mse # NEW
    }

    final_model_name = min(best_mses, key=best_mses.get)
    best_mse_value = best_mses[final_model_name]

    final_model = None
    final_optimizer = None
    final_y_pred_inversed = None
    final_y_test_inversed = None

    if final_model_name == 'RNN':
        final_model, final_optimizer = model_selection_obj.final_model_selection(rnn_optimizer, 'rnn')
        final_y_pred_inversed = rnn_y_pred
        final_y_test_inversed = rnn_y_test
    elif final_model_name == 'LSTM':
        final_model, final_optimizer = model_selection_obj.final_model_selection(lstm_optimizer, 'lstm')
        final_y_pred_inversed = lstm_y_pred
        final_y_test_inversed = lstm_y_test
    elif final_model_name == 'GRU':
        final_model, final_optimizer = model_selection_obj.final_model_selection(gru_optimizer, 'gru')
        final_y_pred_inversed = gru_y_pred
        final_y_test_inversed = gru_y_test
    elif final_model_name == 'XGBoost': # NEW
        final_model, final_optimizer = model_selection_obj.final_model_selection(xgb_optimizer, 'xgboost')
        final_y_pred_inversed = xgb_y_pred
        final_y_test_inversed = xgb_y_test

    if final_model_name:
        st.success(f"The **Best Model** based on lowest MSE is: **{final_model_name}** with MSE: **{best_mse_value:.4f}**")
        st.write("Details of the best model's hyperparameters are logged in MLflow.")
        
        st.subheader(f"Sample of Inversed Values for the Best Model ({final_model_name})")
        st.write("Here are a few sample values of the original target and the best model's predictions, both inverse-transformed:")
        
        if final_y_test_inversed is not None and final_y_pred_inversed is not None:
            final_results_df = pd.DataFrame({
                'Original (Inversed)': final_y_test_inversed.flatten(),
                'Predicted (Inversed)': final_y_pred_inversed.flatten()
            })
            st.dataframe(final_results_df.head(10))
            
            predictions_dir = "predictions"
            os.makedirs(predictions_dir, exist_ok=True)
            predicted_inversed_path = os.path.join(predictions_dir, "predicted_inversed_values.npy")
            
            np.save(predicted_inversed_path, final_y_pred_inversed)
            st.success(f"Predicted inversed values for the best model saved to: `{predicted_inversed_path}`")
            st.session_state['final_y_pred_inversed'] = final_y_pred_inversed
            st.session_state['final_y_test_inversed'] = final_y_test_inversed

        else:
            st.warning("Inversed values for the best model could not be retrieved.")

        st.markdown("### Save Model for Inference")
        with st.spinner("Fitting the final Selected Model..."):
            final_run_id = mlflow_obj.exp_model_selection(final_model,final_model_name,exp_id,final_optimizer,train_X, train_y,test_X, test_y)
            st.info(f"Final MLflow Run ID: {final_run_id}")
        print(final_run_id)
        ret_ml , version , model_name = mlflow_obj.upload_model(final_model_name,final_run_id) 
        if model_name in ["RNN","LSTM","GRU"]:
            versioned_path, latest_path = save_model(final_model, model_name, model_type='keras')
        else:
            versioned_path, latest_path = save_model(final_model, model_name, model_type='xgboost')
        if ret_ml == 0:
            st.success(f"Model Uploaded Successfully name : {model_name} version : {version}")
            print(f"Model Uploaded Successfully name : {model_name} version : {version}")
        else:
            st.error("Model Upload Failed")
            print("Model Upload Failed")

    else:
        st.warning("Could not determine the best model.") 

else:
    st.info("Load data and run training to see visualizations and best model selection.")

st.markdown("---")
st.markdown("For detailed experiment tracking, visit your MLflow UI at: " + MLFLOW_URL)
