# datacenter_dashboard.py
import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
from data_generation_agent import DataCenterPowerAgent # Assuming your class is in this file
import plotly.express as px
import os

# --- Configuration ---
DATA_FILE = "data/historical_datacenter_data.csv" # File to store historical data
HISTORICAL_MAX_ROWS = 5000 # Max rows to keep in memory for historical display
LIVE_MAX_ROWS = 500       # Max rows to keep in memory for live display

st.set_page_config(layout="wide", page_title="Datacenter Dashboard")
st.header("Datacenter Power Consumption Dashboard")

# --- Helper Functions for Data Persistence ---

def load_historical_data():
    """Loads historical data from CSV, or returns an empty DataFrame if file doesn't exist."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, index_col='timestamp', parse_dates=True)
        return df
    return pd.DataFrame()

def save_historical_data(df: pd.DataFrame):
    """Saves the DataFrame to CSV."""
    df.to_csv(DATA_FILE)

def append_to_historical_data(new_df_row: pd.DataFrame):
    """Appends new data to the CSV file without reloading the entire file."""
    if not os.path.exists(DATA_FILE):
        new_df_row.to_csv(DATA_FILE, header=True, index=True)
    else:
        new_df_row.to_csv(DATA_FILE, mode='a', header=False, index=True)

# --- Session State Initialization ---
if 'agent' not in st.session_state:
    # Default parameters for initialization
    default_start_date = "2024-06-01" # Starting data generation from 2024
            
    st.session_state.agent = DataCenterPowerAgent(
        start_date=default_start_date,
        interval_minutes=30, # Data every 30 minutes
        num_total_servers=50,
        num_total_storage_arrays=5,
        num_total_network_devices=10,
        num_total_ups_units=2,
        num_total_lights=30,
        base_outside_temp_celsius=25.0,
        temp_amplitude_daily=5.0,
        temp_amplitude_seasonal=8.0,
        humidity_base_percent=60.0,
        humidity_amplitude_daily=10.0,
        server_idle_power_factor=0.5,
        server_peak_power_kw_per_server=0.35,
        storage_power_kw_per_unit=0.4,
        network_power_kw_per_device=0.1,
        ups_power_kw_per_unit=0.5,
        pdu_efficiency_loss_factor=0.02,
        light_power_kw_per_unit=0.05,
        cooling_base_factor=0.3,
        cooling_temp_sensitivity=0.15,
        cooling_fan_speed_max_percent=100.0,
        cooling_fan_speed_min_percent=30.0,
        # NEWLY ADDED PARAMETERS FOR COOLING SYSTEM
        target_cooling_temp_celsius=22.0,
        cooling_temp_response_factor=0.1,
        base_internal_heat_load_kw=10.0, # Example value, adjust as needed
        cooling_load_factor_temp=0.8,    # Example value, adjust as needed
        cooling_load_factor_humidity=0.1, # Example value, adjust as needed
        workload_cooling_effect_factor=5.0 # Example value, adjust as needed
    )
    # Load historical data at startup
    st.session_state.historical_df = load_historical_data()
    st.session_state.live_df = pd.DataFrame() # Live data will be a subset for immediate display
    st.session_state.running = False

    # Ensure agent's internal time is consistent with the latest historical data
    if not st.session_state.historical_df.empty:
        latest_timestamp = st.session_state.historical_df.index.max()
        # Ensure latest_timestamp is a datetime object
        if isinstance(latest_timestamp, pd.Timestamp):
            latest_timestamp_dt = latest_timestamp.to_pydatetime()
        else: # If it's a string from CSV, parse it
            try:
                latest_timestamp_dt = datetime.strptime(str(latest_timestamp),"%Y-%m-%d %H:%M:%S" )
            except ValueError:
                try:
                    latest_timestamp_dt = datetime.strptime(latest_timestamp,"%Y-%m-%d")
                except Exception as e:
                    st.error(f"Error parsing latest timestamp: {e}")

        st.session_state.agent._current_time = latest_timestamp_dt + timedelta(minutes=st.session_state.agent.interval_minutes)
    else:
        # Assuming DataCenterPowerAgent.start_dt_initial is already a datetime object
        st.session_state.agent._current_time = st.session_state.agent.start_dt_initial

# --- Sidebar Controls ---
st.sidebar.header("Controls")
selected_y_param = st.sidebar.selectbox(
    "Select Parameter for Y-axis:",
    options=[
        'IT power consumption', 'servers_power_kw_internal', 'cooling_power_kw_internal',
        'outside temperature', 'humidity', 'cpu', 'memory', 'fan speed',
        'network devices total power (Kw)', 'UPS_total_power(Kw)', 'PDU_total_power(Kw)',
        'lights_total_power(Kw)', 'servers', 'cooling system ratio', 'cooling temperature' # Added cooling temperature
    ],
    index=0 # Default to IT power consumption
)

update_interval = st.sidebar.slider(
    "Real-time Update Interval (seconds):",
    min_value=1, max_value=10, value=2
)

# Start/Stop Buttons
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("Start Data Generation"):
        st.session_state.running = True
        st.info("Data generation started!") # Use info instead of success initially
with col_btn2:
    if st.button("Stop Data Generation"):
        st.session_state.running = False
        st.warning("Data generation stopped.")

# Reset Data Button
if st.sidebar.button("Reset All Data"):
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE) # Delete the file
    st.session_state.historical_df = pd.DataFrame()
    st.session_state.live_df = pd.DataFrame()
    st.session_state.agent._current_time = st.session_state.agent.start_dt_initial
    st.session_state.running = False
    st.success("All data reset (file deleted). Re-run the app or refresh the browser.")
    st.rerun() # Rerun to re-initialize and generate historical data from scratch

# Download button for all historical data
if not st.session_state.historical_df.empty:
    csv_data = st.session_state.historical_df.to_csv(index=True).encode('utf-8')
    st.sidebar.download_button(
        label="Download All Stored Data (CSV)",
        data=csv_data,
        file_name="all_datacenter_data.csv",
        mime="text/csv",
    )

# --- Tabs Implementation ---
tab_live , tab_historical = st.tabs(["Live Data Feed","Historical Data"])

with tab_historical:
    st.subheader("Historical Data Trends")
    st.write("This tab displays all data collected since the start date, stored on disk.")

    if st.session_state.historical_df.empty:
        st.info("No historical data available yet. Start data generation in 'Live Data Feed' tab.")
    else:
        # Display the full historical data up to a limit for performance
        display_df = st.session_state.historical_df.tail(HISTORICAL_MAX_ROWS)
        
        fig_hist = px.line(display_df, x=display_df.index, y=selected_y_param,
                            title=f'Historical {selected_y_param} Over Time',
                            labels={'index': 'Time', selected_y_param: selected_y_param},
                            height=500)
        fig_hist.update_layout(hovermode="x unified", xaxis_tickformat='%Y-%m-%d %H:%M:%S', uirevision=selected_y_param + "_hist")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Latest Historical Data Points")
        # Ensure timestamp is formatted correctly for display if it's not already
        if 'timestamp' in display_df.columns:
            st.dataframe(display_df.tail(10).style.format(
                {'timestamp': lambda t: t.strftime('%Y-%m-%d %H:%M:%S')}
            ))
        else: # If timestamp is the index
            st.dataframe(display_df.tail(10)) # Pandas handles datetime index formatting well

with tab_live:
    st.subheader("Live Data Stream")
    st.write("This tab updates with real-time generated data.")

    # Placeholders for live chart and table
    live_chart_placeholder = st.empty()
    live_table_placeholder = st.empty()
    status_message_placeholder = st.empty()

    if not st.session_state.running:
        if st.session_state.historical_df.empty:
            status_message_placeholder.info("Click 'Start Data Generation' to begin generating historical data and then live data.")
        else:
            status_message_placeholder.info("Data generation is stopped. Click 'Start Data Generation' to resume live updates.")
            # Display current live_df (if any) even when stopped
            if not st.session_state.live_df.empty:
                with live_chart_placeholder.container():
                    fig_live_stopped = px.line(st.session_state.live_df, x=st.session_state.live_df.index, y=selected_y_param,
                                                 title=f'Live {selected_y_param} (Stopped)',
                                                 labels={'index': 'Time', selected_y_param: selected_y_param},
                                                 height=500)
                    fig_live_stopped.update_layout(hovermode="x unified", xaxis_tickformat='%Y-%m-%d %H:%M:%S', uirevision=selected_y_param + "_live")
                    st.plotly_chart(fig_live_stopped, use_container_width=True)
                with live_table_placeholder.container():
                    st.subheader("Latest Live Data Points (Stopped)")
                    st.dataframe(st.session_state.live_df.tail(10)) # Pandas handles datetime index formatting well

    # --- Data Generation Loop (only runs when 'running' is True) ---
    if st.session_state.running:
        status_message_placeholder.success("Generating live data...")
        # Check if historical data needs to be caught up first
        if st.session_state.agent._current_time < datetime.now():
            status_message_placeholder.info("Catching up on historical data... please wait.")
            while st.session_state.agent._current_time < datetime.now():
                new_data_point = st.session_state.agent.generate_single_data_point()
                new_df_row = pd.DataFrame([new_data_point]).set_index('timestamp')
                
                # Append to the historical DataFrame and save to disk
                st.session_state.historical_df = pd.concat([st.session_state.historical_df, new_df_row])
                append_to_historical_data(new_df_row) # Save to disk immediately

                # Also add to live_df for immediate display (if it's recent enough)
                st.session_state.live_df = pd.concat([st.session_state.live_df, new_df_row])
                if len(st.session_state.live_df) > LIVE_MAX_ROWS:
                    st.session_state.live_df = st.session_state.live_df.tail(LIVE_MAX_ROWS)

                # Update UI periodically during catch-up for responsiveness
                # Update every few simulated minutes, or more often if interval_minutes is large
                if (st.session_state.agent._current_time.minute % (st.session_state.agent.interval_minutes * 2) == 0): 
                     with live_chart_placeholder.container():
                        if not st.session_state.live_df.empty:
                            fig_live_catchup = px.line(st.session_state.live_df, x=st.session_state.live_df.index, y=selected_y_param,
                                                         title=f'Live {selected_y_param} (Catching Up)',
                                                         labels={'index': 'Time', selected_y_param: selected_y_param},
                                                         height=500)
                            fig_live_catchup.update_layout(hovermode="x unified", xaxis_tickformat='%Y-%m-%d %H:%M:%S', uirevision=selected_y_param + "_live")
                            st.plotly_chart(fig_live_catchup, use_container_width=True)
                     with live_table_placeholder.container():
                        st.subheader("Latest Real-time Data Points (Catching Up)")
                        st.dataframe(st.session_state.live_df.tail(10))
                     time.sleep(0.01) # Small sleep to yield to Streamlit
            status_message_placeholder.success("Historical catch-up complete. Now generating live data.")
            st.session_state.live_df = pd.DataFrame() # Clear live_df to start fresh for true live data
            st.rerun() # Rerun to clear any old plots from the catch-up phase and ensure it draws clean for live data

        # --- Main Real-time Data Loop ---
        while st.session_state.running:
            new_data_point = st.session_state.agent.generate_single_data_point()
            new_df_row = pd.DataFrame([new_data_point]).set_index('timestamp')
            
            # Append to in-memory DataFrames
            st.session_state.historical_df = pd.concat([st.session_state.historical_df, new_df_row])
            st.session_state.live_df = pd.concat([st.session_state.live_df, new_df_row])

            # Persist the new data point to disk
            append_to_historical_data(new_df_row)

            # Limit the size of the live_df for performance
            if len(st.session_state.live_df) > LIVE_MAX_ROWS:
                st.session_state.live_df = st.session_state.live_df.tail(LIVE_MAX_ROWS)
            
            # Limit the size of historical_df in memory, if needed for extremely long runs
            if len(st.session_state.historical_df) > HISTORICAL_MAX_ROWS * 2: # Keep more in memory for smoothing
                    st.session_state.historical_df = st.session_state.historical_df.tail(HISTORICAL_MAX_ROWS * 2)

            # Update UI
            with live_chart_placeholder.container():
                if not st.session_state.live_df.empty:
                    fig_live = px.line(st.session_state.live_df, x=st.session_state.live_df.index, y=selected_y_param,
                                         title=f'Live {selected_y_param} Over Time',
                                         labels={'index': 'Time', selected_y_param: selected_y_param},
                                         height=500)
                    fig_live.update_layout(hovermode="x unified", xaxis_tickformat='%Y-%m-%d %H:%M:%S', uirevision=selected_y_param + "_live")
                    st.plotly_chart(fig_live, use_container_width=True)

            with live_table_placeholder.container():
                st.subheader("Latest Real-time Data Points")
                st.dataframe(st.session_state.live_df.tail(10)) # Pandas handles datetime index formatting well

            time.sleep(update_interval)

        # After the loop breaks (stopped), ensure the status is clear
        status_message_placeholder.empty() # Clear the "Generating live data..." message
        st.experimental_rerun() # This is critical to ensure Streamlit re-renders the UI after stopping