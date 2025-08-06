import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import os
import joblib
import warnings
import tensorflow as tf

from mlops import load_latest_model
from data_generation_agent import DataCenterPowerAgent
from coolingagent import DataCenterCoolingAgent

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide")
st.title("📈 AI Enabled Data Center Power and PUE Dashboard")
#st.markdown("---")

warnings.filterwarnings("ignore", category=UserWarning)

# --- Constants and Data Paths ---
HISTORICAL_DATA_PATH = "data/historical_datacenter_data.csv"
OPTIMIZED_DATA = "data/inference_data/dc_power_optimized.csv"
REGULAR_DATA = "data/inference_data/dc_power_regular.csv"
LOOK_BACK_WINDOW = 48
#MODEL_PATH = "C:/Users/guptav31/mlflow-run/new/31_07_2025_17_18/1dfca82d175746e2b5aa15b5cce41360/artifacts/Datacenter_lstm/data/model.keras"
MODEL_PATH = "C:/Users/guptav31/mlflow-run/new/02_08_2025_21_02/9404c7d1844540719e81c5387d3819de/artifacts/Datacenter_lstm/data/model.keras"
SCALER_PATH = "scaler_data/scaler_y.joblib"
TEST_DATA_PATH = 'preprocessed_data/test_X.npy'


# --- Mock Scaler Class ---
class MockScaler:
    "A mock scaler to simulate the behavior of a real scikit-learn scaler."
    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.fitted = False

    def fit(self, data):
        data = np.array(data, dtype=np.float64)
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)
        self.fitted = True
        return self

    def transform(self, data):
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
        data = np.array(data, dtype=np.float64)
        return (data - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, data):
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
        data = np.array(data, dtype=np.float64)
        return data * (self.max_val - self.min_val) + self.min_val


# --- Helper Function to Load Scaler ---
def load_scaler(scaler_path):
    """
    Attempts to load a scaler object from a local file.
    Returns a mock scaler if the file is not found.
    """
    try:
        scaler = joblib.load(scaler_path)
        st.success(f"Loading scaler from local file")
        return scaler
    except FileNotFoundError:
        st.warning(f"Scaler file not found at '{scaler_path}'. Using a mock scaler instead.")
        return MockScaler()
    except Exception as e:
        st.error(f"Error loading scaler: {e}. Using a mock scaler.")
        return MockScaler()


def optimized_data(historical_df, prediction_start_date, prediction_duration_days,model_name):
    # --- Load Model and Scaler ---
    
    with st.spinner("Loading model and scaler for optimized scenario..."):
        try:
            #prediction_model = tf.keras.models.load_model(MODEL_PATH)
            if model_name in ["RNN","LSTM","GRU"]:
                model_type = 'keras'
            else:
                model_type = 'xgboost'
            prediction_model = load_latest_model(model_name,model_type)
            if prediction_model:
                st.success("Model loaded successfully")
            else:
                st.error("Failed to load model, Please re-check your model name")
                st.stop()
        except Exception as e:
            st.error(f"Failed to load model from '{MODEL_PATH}': {e}")
            st.stop()
        
        scaler = load_scaler(SCALER_PATH)

    try:
        test_X = np.load(TEST_DATA_PATH, allow_pickle=True)
        global LOOK_BACK_WINDOW
        LOOK_BACK_WINDOW = test_X.shape[1]
    except FileNotFoundError:
        st.error(f"Test data file not found at '{TEST_DATA_PATH}'. Prediction requires preprocessed data.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        st.stop()
    
    # --- Define Feature Columns ---
    feature_cols = [
        'outside temperature', 'humidity', 'servers', 'cpu', 'memory',
        'num network devices', 'no of UPS', 'no of light', 'servers_power_kw_internal',
        'storage_total_power_kw', 'network devices total power (Kw)', 'IT power consumption',
        'UPS_total_power(Kw)', 'PDU_total_power(Kw)', 'lights_total_power(Kw)',
        'cooling_power_kw_internal', 'cooling system ratio', 'fan speed'
    ]
    
    prediction_col = 'IT power consumption'
    NUM_FEATURES = len(feature_cols)

    # Ensure all required columns are in the historical data
    missing_cols = [col for col in feature_cols if col not in historical_df.columns]
    if missing_cols:
        st.error(f"The following required feature columns are missing from the dataset: {', '.join(missing_cols)}")
        st.stop()
    

    # --- Generate Future Predictions ---
    prediction_points = prediction_duration_days * 24 * (60 // 30)
    
    last_sequence = test_X[-1]
    look_back_data = last_sequence.reshape(1, LOOK_BACK_WINDOW, NUM_FEATURES)
    
    predicted_scaled_values = []
    
    with st.spinner(f"Predicting {prediction_points} data points for the next {prediction_duration_days} days..."):
        for _ in range(prediction_points):
            next_prediction_scaled = prediction_model.predict(look_back_data, verbose=0)
            predicted_scaled_values.append(next_prediction_scaled.flatten()[0])
            new_row_scaled = look_back_data[0, -1].copy()
            pred_col_index = feature_cols.index(prediction_col)
            new_row_scaled[pred_col_index] = next_prediction_scaled[0][0]
            new_look_back_data = np.append(look_back_data[0, 1:], new_row_scaled).reshape(1, LOOK_BACK_WINDOW, NUM_FEATURES)
            look_back_data = new_look_back_data

    dummy_array_for_inverse = np.zeros((len(predicted_scaled_values), NUM_FEATURES))
    pred_col_index = feature_cols.index(prediction_col)
    dummy_array_for_inverse[:, pred_col_index] = predicted_scaled_values
    
    predicted_values_all_features = scaler.inverse_transform(dummy_array_for_inverse)
    predicted_values = predicted_values_all_features[:, pred_col_index]
    
    #last_historical_timestamp_str = historical_df.index[-1]
    #last_historical_timestamp = pd.to_datetime(last_historical_timestamp_str)
    #prediction_start_date = last_historical_timestamp + timedelta(minutes=30)
    future_timestamps = [prediction_start_date + timedelta(minutes=30 * i) for i in range(prediction_points)]
    
    #predicted_df = pd.DataFrame(data={prediction_col: predicted_values}, index=future_timestamps)

    cooling_agent = DataCenterCoolingAgent(
        start_date=prediction_start_date.strftime("%Y-%m-%d"),
        interval_minutes=30,
        num_total_servers=50, num_total_storage_arrays=5, num_total_network_devices=10,
        num_total_ups_units=2, num_total_lights=30,
        base_outside_temp_celsius=25.0, temp_amplitude_daily=5.0, temp_amplitude_seasonal=8.0,
        humidity_base_percent=60.0, humidity_amplitude_daily=10.0,
        server_idle_power_factor=0.5, server_peak_power_kw_per_server=0.35,
        storage_power_kw_per_unit=0.4, network_power_kw_per_device=0.1,
        ups_power_kw_per_unit=0.3, pdu_efficiency_loss_factor=0.01,
        light_power_kw_per_unit=0.03, cooling_base_factor=0.2,
        cooling_temp_sensitivity=0.15, cooling_fan_speed_max_percent=100.0,
        cooling_fan_speed_min_percent=30.0, target_cooling_temp_celsius=25.0,
        cooling_temp_response_factor=0.1, base_internal_heat_load_kw=10.0,
        cooling_load_factor_temp=0.8, cooling_load_factor_humidity=0.1,
        workload_cooling_effect_factor=5.0, cooling_system_cop=4.0
    )
    predicted_datacenter_data = []
    for i, each_predicted_value in enumerate(predicted_values):
        predicted_it_power = each_predicted_value
        timestamp = future_timestamps[i]
        data_point = cooling_agent.generate_single_data_point(
            it_power_consumption=predicted_it_power,
            timestamp_to_generate=timestamp
        )
        predicted_datacenter_data.append(data_point)

    df_predicted = pd.DataFrame(predicted_datacenter_data)
    df_predicted.set_index('timestamp', inplace=True)
    df_predicted.to_csv(OPTIMIZED_DATA)
    if df_predicted.empty:
        st.error("Predicted data is empty. Cannot perform predictions.")
        st.stop()
    else:
        #st.success("Predictions generated successfully.")
        st.success("Optimized data generated successfully.")
        #st.balloons()
    return df_predicted


def regular_data(prediction_start_date, prediction_duration_days):
    #last_historical_timestamp_str = historical_df.index[-1]
    #last_historical_timestamp = pd.to_datetime(last_historical_timestamp_str)
    #prediction_start_date = last_historical_timestamp + timedelta(minutes=30)
    
    prediction_points = prediction_duration_days * 24 * (60 // 30)
    before_df = pd.DataFrame()
    
    with st.spinner(f"Generating {prediction_points} data points for the next {prediction_duration_days} days..."):
        agent = DataCenterPowerAgent(
            start_date=prediction_start_date.strftime("%Y-%m-%d"),
            interval_minutes=30,
            num_total_servers=50, num_total_storage_arrays=5, num_total_network_devices=10,
            num_total_ups_units=2, num_total_lights=30,
            base_outside_temp_celsius=25.0, temp_amplitude_daily=5.0, temp_amplitude_seasonal=8.0,
            humidity_base_percent=60.0, humidity_amplitude_daily=10.0,
            server_idle_power_factor=0.5, server_peak_power_kw_per_server=0.35,
            storage_power_kw_per_unit=0.4, network_power_kw_per_device=0.1,
            ups_power_kw_per_unit=0.3, pdu_efficiency_loss_factor=0.01,
            light_power_kw_per_unit=0.03, cooling_base_factor=0.2,
            cooling_temp_sensitivity=0.15, cooling_fan_speed_max_percent=100.0,
            cooling_fan_speed_min_percent=30.0, target_cooling_temp_celsius=25.0,
            cooling_temp_response_factor=0.1, base_internal_heat_load_kw=10.0,
            cooling_load_factor_temp=0.8, cooling_load_factor_humidity=0.1,
            workload_cooling_effect_factor=5.0, cooling_system_cop=4.0
        )
        
        for _ in range(prediction_points):
            data_point = agent.generate_single_data_point()
            df_row = pd.DataFrame([data_point]).set_index('timestamp')
            before_df = pd.concat([before_df, df_row])
    before_df.to_csv(REGULAR_DATA)
    if before_df.empty:
        st.error("No data points generated. Please try again.")
        st.stop()
    else:
        st.success(f"Regular data generated successfully. Generated data points for the next {prediction_duration_days} days.")
        #st.balloons()
    return before_df

def main(prediction_duration_days,model_name):

    # --- Load Historical Data ---
    try:
        if not os.path.exists(HISTORICAL_DATA_PATH):
            st.error(f"File not found: {HISTORICAL_DATA_PATH}. Please ensure the file is in the correct directory.")
            st.stop()
        historical_df = pd.read_csv(HISTORICAL_DATA_PATH, index_col='timestamp', parse_dates=True)
        if historical_df.empty:
            st.error("Historical data file is empty. Cannot perform predictions.")
            st.stop()
        st.info(f"Loaded historical data from '{HISTORICAL_DATA_PATH}'")
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        st.stop()

    last_historical_timestamp_str = historical_df.index[-1]
    last_historical_timestamp = pd.to_datetime(last_historical_timestamp_str)
    prediction_start_date = last_historical_timestamp + timedelta(minutes=30)
    #future_timestamps = [prediction_start_date + timedelta(minutes=30 * i) for i in range(prediction_points)]

    st.header("Normal Datacenter and AI enabled Datacenter data generation results")
    st.markdown("---")
    
    # Generate and process data for both scenarios
    st.subheader("Regular Data Generation")
    with st.spinner("Generating data for 'Regular' Datacenter data..."):
        before_df = regular_data( prediction_start_date,prediction_duration_days)
    st.subheader("Optimized Data Generation")
    with st.spinner("Generating data for 'Optimized' Datacenter data..."):
        optimized_df = optimized_data(historical_df,prediction_start_date ,prediction_duration_days,model_name)

    regular_calculated_df = before_df.copy()
    regular_calculated_df['total_power_consumption'] = (
        regular_calculated_df['IT power consumption'] +
        regular_calculated_df['UPS_total_power(Kw)'] +
        regular_calculated_df['PDU_total_power(Kw)'] +
        regular_calculated_df['lights_total_power(Kw)'] +
        regular_calculated_df['cooling_power_kw_internal']
    )
    regular_calculated_df['PUE'] = regular_calculated_df.apply(
        lambda row: row['total_power_consumption'] / row['IT power consumption'] 
        if row['IT power consumption'] != 0 else np.nan, axis=1
    )
    regular_total_power_sum = regular_calculated_df['total_power_consumption'].sum() / 2
    regular_average_pue = regular_calculated_df['PUE'].mean()

    optimized_calculated_df = optimized_df.copy()
    optimized_calculated_df['total_power_consumption'] = (
        optimized_calculated_df['IT power consumption'] +
        optimized_calculated_df['UPS_total_power(Kw)'] +
        optimized_calculated_df['PDU_total_power(Kw)'] +
        optimized_calculated_df['lights_total_power(Kw)'] +
        optimized_calculated_df['cooling_power_kw_internal']
    )
    
    optimized_calculated_df['PUE'] = optimized_calculated_df.apply(
        lambda row: row['total_power_consumption'] / row['IT power consumption'] 
        if row['IT power consumption'] != 0 else np.nan, axis=1
    )
    
    #inter_df = pd.concat([optimized_calculated_df['total_power_consumption'], regular_calculated_df['IT power consumption']],axis=1)
    #optimized_calculated_df['PUE'] = inter_df.apply(
    #lambda row: row['total_power_consumption'] / row['IT power consumption']
    #if row['IT power consumption'] != 0 else np.nan, axis=1
    #)
    optimized_total_power_sum = optimized_calculated_df['total_power_consumption'].sum() / 2
    optimized_average_pue = optimized_calculated_df['PUE'].mean()
    
    power_savings = regular_total_power_sum - optimized_total_power_sum
    power_savings_percent = (power_savings / regular_total_power_sum) * 100
    pue_improvement = regular_average_pue - optimized_average_pue


    # --- Display Key Performance Indicators ---
    st.header("Key Performance Indicators")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="Total Facility Power (Before)", value=f"{int(regular_total_power_sum):,} kWh")
    with col2:
        st.metric(label="Total Facility Power (After)", value=f"{int(optimized_total_power_sum):,} kWh") #,.2f
    with col3:
        st.metric(label="Power Savings", value=f"{int(power_savings):,} kWh", delta=f"{power_savings_percent:.2f}%")
    with col4:
        st.metric(label="Average PUE (Before)", value=f"{regular_average_pue:.2f}")
    with col5:
        st.metric(label="Average PUE (After)", value=f"{optimized_average_pue:.2f}", delta=f"{-pue_improvement:.2f}")

    st.markdown("---")
    st.balloons()
    
    # --- Plot Power Consumption Over Time ---
    st.header("Power Consumption Over Time")
    st.markdown("Total Facility Power Consumption Over Time (Before vs. After)")

    combined_power_df = pd.DataFrame({
        'Power Before (kW)': regular_calculated_df['total_power_consumption'],
        'Power After (kW)': optimized_calculated_df['total_power_consumption']
    })

    power_fig = px.line(
        combined_power_df,
        y=['Power Before (kW)', 'Power After (kW)'],
        labels={'value': 'Total Power Consumption (kW)', 'variable': 'Scenario'},
        color_discrete_map={'Power Before (kW)': 'blue', 'Power After (kW)': 'skyblue'}
    )
    st.plotly_chart(power_fig, use_container_width=True)
    
    # Add PUE graph here
    # --- Plot PUE Over Time ---
    st.markdown("---")
    st.header("Power Usage Effectiveness (PUE) Over Time")
    st.markdown("PUE Over Time (Before vs. After)")
    
    combined_pue_df = pd.DataFrame({
        'PUE Before': regular_calculated_df['PUE'],
        'PUE After': optimized_calculated_df['PUE']
    })
    
    pue_fig = px.line(
        combined_pue_df,
        y=['PUE Before', 'PUE After'],
        labels={'value': 'PUE Ratio', 'variable': 'Scenario'},
        color_discrete_map={'PUE Before': 'red', 'PUE After': 'lightcoral'}
    )
    st.plotly_chart(pue_fig, use_container_width=True)

if __name__ == "__main__":
    with st.form(key='my_form'):
    
        st.subheader("Prediction Settings")
        prediction_duration_days = st.slider(
            "Select Prediction Duration (Days)",
            min_value=1,
            max_value=30,
            value=7,
            step=1
        )
        dropdown_options = ["LSTM", "RNN", "GRU", "XGBoost"]
        model_name = st.selectbox("Select Model", options=dropdown_options,
        index=0  # Default to the first option
        )
    
    # The submit_button will return True when clicked
        submit_button = st.form_submit_button(label='Submit and Start Process')

# After the form is submitted, the code inside this block will run
    if submit_button:
        main(prediction_duration_days,model_name)
