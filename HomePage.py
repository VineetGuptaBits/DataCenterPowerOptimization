import streamlit as st

st.set_page_config(
    page_title="home",
    page_icon="🏠",
    layout="wide"
)

st.title("Datacenter Dashboard App!")

st.markdown(
    """
    Welcome to Datacenter Power Usage Prediction Dashboard.
    1. DataGeneration Page is to Simulate the generation of Historical Data as well as generation of real-time data.
    2. DataAnalysis Page is to perform EDA on the data.
    """
)
