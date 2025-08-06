import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import correlate
#from prophet import Prophet
#from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import plotly.graph_objects as go

# Set Matplotlib style for better aesthetics
plt.style.use('ggplot')

# --- Page Configuration ---
st.set_page_config(
    page_title="DataAnalysis ",
    page_icon="💡",
    layout="wide"
)

# --- Title and Description ---
st.title("💡 Exploratory Data Analysis of Datacenter Power consumption data")
st.markdown("""
This application helps you perform Exploratory Data Analysis (EDA) on your multivariate time series data

**Instructions:**
1. CSV file will be automatically fetched from DataGenerator. Ensure it contains a column that can be parsed as a date/time.
2. Select the date/time column and any other columns you want to include in the analysis.
3. Explore the various EDA sections.

""")

def run_adfuller_test(series):
    result = adfuller(series.dropna())
    st.write(f'ADF Statistic: {result[0]:.2f}')
    st.write(f'p-value: {result[1]:.3f}')
    st.write('Critical Values:')
    for key, value in result[4].items():
        st.write(f'   {key}: {value:.3f}')
    if result[1] <= 0.05:
        st.success("Conclusion: The series is likely stationary (reject H0).")
    else:
        st.warning("Conclusion: The series is likely non-stationary (fail to reject H0).")

def run_kpss_test(series):
    result = kpss(series.dropna())
    st.write(f'KPSS Statistic: {result[0]:.2f}')
    st.write(f'p-value: {result[1]:.3f}')
    st.write('Critical Values:')
    for key, value in result[3].items():
        st.write(f'   {key}: {value:.3f}')
    if result[1] <= 0.05:
        st.warning("Conclusion: The series is likely non-stationary (reject H0).")
    else:
        st.success("Conclusion: The series is likely stationary (fail to reject H0).")



def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(series, threshold=3):
    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std
    outliers = series[np.abs(z_scores) > threshold]
    return outliers, threshold


#st.sidebar.header("Reading Your Data")
uploaded_file = 'data/historical_datacenter_data.csv'

df = None  # Initialize df outside the if block

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Reading file was successfully!")
        #st.sidebar.dataframe(df.head()) # Show a glimpse of the uploaded data in sidebar
    except Exception as e:
        st.error(f"Error reading file: {e}. Please ensure it's a valid CSV.")
else:
    st.info("Please upload a CSV file to begin your EDA.")

# --- EDA Sections (only if DataFrame is loaded) ---
if df is not None:
    st.header("1. Data Overview")

    # Display raw data
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data (First 5 Rows)")
        st.dataframe(df.head())

    # Display DataFrame info
    st.subheader("DataFrame Information")
    buffer = pd.io.common.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T) # Transpose for better readability


    # --- Missing Values Analysis & Imputation ---
    st.header("2. Missing Value Analysis & Imputation")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

    if not missing_data.empty:
        st.write("Columns with Missing Values:")
        st.dataframe(missing_data.rename('Missing Count').to_frame())

        st.subheader("Missing Value Percentage")
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
        st.dataframe(missing_percentage.rename('Missing Percentage (%)').to_frame())

        st.subheader("Missing Value Heatmap")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
        plt.title('Missing Values Heatmap')
        st.pyplot(fig)

        st.subheader("Imputation Options")
        imputation_method = st.radio(
            "Select Imputation Method:",
            ("None (keep NaNs)", "Forward Fill (ffill)", "Backward Fill (bfill)", "Mean Imputation", "Median Imputation")
        )

        if imputation_method == "Forward Fill (ffill)":
            df_imputed = df.ffill()
        elif imputation_method == "Backward Fill (bfill)":
            df_imputed = df.bfill()
        elif imputation_method == "Mean Imputation":
            df_imputed = df.fillna(df.mean(numeric_only=True))
        elif imputation_method == "Median Imputation":
            df_imputed = df.fillna(df.median(numeric_only=True))
        else:
            df_imputed = df.copy() # No imputation

        if imputation_method != "None (keep NaNs)":
            st.write(f"DataFrame after '{imputation_method}' imputation:")
            st.dataframe(df_imputed.head())
            st.write("Missing values after imputation:")
            st.dataframe(df_imputed.isnull().sum()[df_imputed.isnull().sum() > 0].to_frame('Remaining Missing'))
            df = df_imputed # Use imputed df for further analysis
    else:
        st.info("No missing values found in the dataset.")
        st.dataframe(df.isnull().sum().T)
        df_imputed = df.copy() # Ensure df_imputed is defined even if no NaNs

    st.markdown("---")
    numerical_cols = []
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # --- Outlier Detection and Visualization ---
    st.header("3. Outlier Detection and Visualization")
    if numerical_cols:
        outlier_col = st.selectbox("Select a numerical column to check for outliers:", numerical_cols, key='outlier_select')

        if outlier_col:
            st.subheader(f"Outlier Analysis for: {outlier_col}")

            outlier_method = st.radio(
                "Select Outlier Detection Method:",
                ("IQR Method", "Z-score Method")
            )

            outliers_found = pd.Series([], dtype=df[outlier_col].dtype) # Initialize empty series
            num_outliers = 0

            if outlier_method == "IQR Method":
                outliers, lower_bound, upper_bound = detect_outliers_iqr(df[outlier_col].dropna())
                num_outliers = len(outliers)
                st.write(f"**IQR Method Results:**")
                st.write(f"Lower Bound: {lower_bound:.2f}")
                st.write(f"Upper Bound: {upper_bound:.2f}")
                if num_outliers > 0:
                    st.write(f"Number of outliers detected: {num_outliers}")
                    st.dataframe(outliers.to_frame('Outlier Value'))
                    outliers_found = outliers
                else:
                    st.info("No outliers detected using the IQR method.")

                st.subheader("Box Plot for Outlier Visualization (IQR)")
                fig_box = px.box(df, y=outlier_col, title=f'Box Plot of {outlier_col} (IQR Outliers)')
                st.plotly_chart(fig_box, use_container_width=True)

            elif outlier_method == "Z-score Method":
                z_score_threshold = st.slider("Z-score Threshold:", 1.0, 4.0, 3.0, 0.1)
                outliers, threshold = detect_outliers_zscore(df[outlier_col].dropna(), threshold=z_score_threshold)
                num_outliers = len(outliers)
                st.write(f"**Z-score Method Results (Threshold: {threshold}):**")
                if num_outliers > 0:
                    st.write(f"Number of outliers detected: {num_outliers}")
                    st.dataframe(outliers.to_frame('Outlier Value'))
                    outliers_found = outliers
                else:
                    st.info("No outliers detected using the Z-score method.")

            # Plot time series with outliers highlighted (for both methods)
            if num_outliers > 0:
                st.subheader(f"Time Series Plot of {outlier_col} with Outliers Highlighted")
                plot_df = df[[outlier_col]].copy()
                plot_df['Outlier'] = False
                plot_df.loc[outliers_found.index, 'Outlier'] = True

                fig = px.line(plot_df, x=plot_df.index, y=outlier_col,
                              title=f'Time Series Plot of {outlier_col} with Outliers',
                              labels={'x': 'Date/Time', 'y': outlier_col})

                # Add scatter points for outliers
                outlier_points = plot_df[plot_df['Outlier']]
                fig.add_trace(go.Scatter(x=outlier_points.index, y=outlier_points[outlier_col],
                                         mode='markers',
                                         name='Outliers',
                                         marker=dict(color='red', size=8, symbol='star')))
                fig.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
    else:
        st.info("No numerical columns available for outlier detection.")

        # --- Time Series Plots ---
    
    st.header("4. Time Series Visualizations")

    if numerical_cols:
        st.subheader("Individual Time Series Plots")
        selected_single_series = st.selectbox(
            "Select a series to plot its trend:", numerical_cols, key='single_plot_select'
        )
        if selected_single_series:
            fig = px.line(df, x=df.index, y=selected_single_series,
                          title=f'Time Series Plot of {selected_single_series}',
                          labels={'x': 'Date/Time', 'y': selected_single_series})
            fig.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Overlayed Time Series Plots")
        selected_overlay_series = st.multiselect(
            "Select series to overlay (max 5 recommended):", numerical_cols,
            default=numerical_cols[:min(len(numerical_cols), 3)] # Default to first 3 or fewer
        )
        if selected_overlay_series:
            fig = px.line(df, x=df.index, y=selected_overlay_series,
                          title='Overlayed Time Series Plots',
                          labels={'x': 'Date/Time', 'value': 'Value'})
            fig.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numerical columns found for time series plots.")
    st.markdown("---")

    # --- Correlation Analysis ---
    st.header("5. Correlation Analysis")
    if numerical_cols and len(numerical_cols) > 1:
        st.subheader("Correlation Heatmap (Overall)")
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        plt.title('Correlation Heatmap of Numerical Features')
        st.pyplot(fig)

        st.subheader("Cross-Correlation Plots (Lagged)")
        st.info("Cross-correlation shows how two series are correlated at different time lags. "
                "A positive peak at a positive lag means the first series leads the second.")

        col_x = st.selectbox("Select Series X:", numerical_cols, key='ccf_x')
        col_y = st.selectbox("Select Series Y:", numerical_cols, key='ccf_y')
        max_lags = st.slider("Max Lags for Cross-Correlation:", 5, 50, 10)

        if col_x and col_y and col_x != col_y:
            series1 = df[col_x].dropna()
            series2 = df[col_y].dropna()

            if len(series1) == 0 or len(series2) == 0:
                st.warning("One or both selected series are empty after dropping NaNs. Cannot compute cross-correlation.")
            else:
                # Align series based on common index
                common_index = series1.index.intersection(series2.index)
                series1_aligned = series1.loc[common_index]
                series2_aligned = series2.loc[common_index]

                if len(series1_aligned) < 2 * max_lags:
                    st.warning(f"Not enough data points ({len(series1_aligned)}) to compute cross-correlation for {max_lags} lags. Consider reducing max lags or using more data.")
                else:
                    cross_corr = correlate(series1_aligned - series1_aligned.mean(),
                                           series2_aligned - series2_aligned.mean(),
                                           mode='full') / (len(series1_aligned) * series1_aligned.std() * series2_aligned.std())
                    lags = np.arange(-len(series1_aligned) + 1, len(series1_aligned))
                    # Filter lags to display
                    lags_to_plot = lags[(lags >= -max_lags) & (lags <= max_lags)]
                    cross_corr_to_plot = cross_corr[(lags >= -max_lags) & (lags <= max_lags)]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.stem(lags_to_plot, cross_corr_to_plot, basefmt=" ")
                    ax.set_title(f'Cross-Correlation between {col_x} and {col_y}')
                    ax.set_xlabel('Lag (Series Y relative to Series X)')
                    ax.set_ylabel('Correlation')
                    st.pyplot(fig)
                    st.write(f"Interpretation: Positive lag (e.g., +5) means {col_x} leads {col_y} by 5 time units. Negative lag (-5) means {col_y} leads {col_x} by 5 time units.")
        elif col_x == col_y:
            st.info("Please select two different series for cross-correlation.")
    else:
        st.info("Not enough numerical columns to perform correlation analysis.")
    st.markdown("---")

    # --- Autocorrelation and Stationarity ---
    st.header("6. Autocorrelation & Stationarity")
    if numerical_cols:
        selected_series_for_acf = st.selectbox(
            "Select a series for ACF/PACF and Stationarity Tests:", numerical_cols, key='acf_pacf_select'
        )
        if selected_series_for_acf:
            st.subheader(f"ACF and PACF Plots for {selected_series_for_acf}")
            fig_acf, ax_acf = plt.subplots(figsize=(10, 5))
            plot_acf(df[selected_series_for_acf].dropna(), lags=40, ax=ax_acf, title=f'ACF of {selected_series_for_acf}')
            st.pyplot(fig_acf)

            fig_pacf, ax_pacf = plt.subplots(figsize=(10, 5))
            plot_pacf(df[selected_series_for_acf].dropna(), lags=40, ax=ax_pacf, title=f'PACF of {selected_series_for_acf}')
            st.pyplot(fig_pacf)

            st.subheader(f"Stationarity Tests for {selected_series_for_acf}")
            st.markdown("--- **Augmented Dickey-Fuller (ADF) Test** ---")
            run_adfuller_test(df[selected_series_for_acf])

            st.markdown("--- **Kwiatkowski–Phillips–Schmidt–Shin (KPSS) Test** ---")
            run_kpss_test(df[selected_series_for_acf])

            st.info("Remember: ADF tests for the presence of a unit root (non-stationarity as null hypothesis). KPSS tests for stationarity around a deterministic trend (stationarity as null hypothesis).")
    st.markdown("---")

