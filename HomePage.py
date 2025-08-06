import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="home",
    page_icon="🏠",
    layout="wide"
)

st.title("Datacenter Power Usage Prediction and Optimization Dashboard")
#st.session_state = st.session_state if 'session_state' in st.session_state else None
st.markdown("**Welcome Onboard...!**")

col1, col2 = st.columns([.2, 1])
with col1:
    st.page_link("pages/1-DataGeneration.py", label="DataGeneration" , icon="📊")
    st.page_link("pages/2-DataAnalysis.py", label="DataAnalysis" , icon="💡")
    st.page_link("pages/3-DataPreprocessing.py", label="DataPreprocessing" , icon="⚙️")
    st.page_link("pages/4-ModelTrainOptimize.py", label="ModelTrainOptimize" , icon="🧠" )
    st.page_link("pages/5-Inference.py", label="Inference" , icon="📈")
with col2:
    st.markdown("is to Simulate the generation of Historical Data as well as generation of real-time data.")
    st.markdown("is to perform EDA on the data.")
    st.markdown("is to preprocess the data for the model training.")
    st.markdown("is to train, optimize and evaluate the model.")
    st.markdown("is to perform inference on the trained model. And the dashborad will show the Key performance metrics.")

historical_total_df = pd.read_csv("data/historical_datacenter_data.csv")
total_power_df = historical_total_df.copy()

total_power_df['Total power consumption'] = (
        total_power_df['IT power consumption'] +
        total_power_df['UPS_total_power(Kw)'] +
        total_power_df['PDU_total_power(Kw)'] +
        total_power_df['lights_total_power(Kw)'] +
        total_power_df['cooling_power_kw_internal']
    )
total_power_df['PUE'] = total_power_df.apply(
        lambda row: row[['Total power consumption']] / row['IT power consumption'] 
        if row['IT power consumption'] != 0 else np.nan, axis=1
    )
total_power_sum = total_power_df[['Total power consumption']].sum() / 2
average_pue = total_power_df['PUE'].mean()
st.header("Historical data Intuition")
st.markdown("---")

col3, col4 = st.columns(2)
    
with col3:
    st.metric(label="Current Total Facility Power consumed", value=f"{int(total_power_sum):,} kWh")
with col4:
    st.metric(label="Average PUE (Before)", value=f"{average_pue:.2f}")
st.markdown("---")

selected_y_param = st.sidebar.selectbox(
    "Select Parameter for Y-axis:",
    options=[
        'Total power consumption', 'IT power consumption','PUE' # Added cooling temperature
    ],
    index=0 # Default to IT power consumption
)

HISTORICAL_MAX_ROWS = 1000

#st.session_state.total_power_df = total_power_df
display_df = total_power_df.tail(HISTORICAL_MAX_ROWS)
        
fig_hist = px.line(display_df, x=display_df.index, y=selected_y_param,
                            title=f'Historical {selected_y_param} Over Time',
                            labels={'index': 'Time', selected_y_param: selected_y_param},
                            height=500)
fig_hist.update_layout(hovermode="x unified", xaxis_tickformat='%Y-%m-%d %H:%M:%S', uirevision=selected_y_param + "_hist")
st.plotly_chart(fig_hist, use_container_width=True)

