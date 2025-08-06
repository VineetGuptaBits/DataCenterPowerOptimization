import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
import os
import joblib # For saving and loading scaler objects

class Preprocessing:
    """
    A class to perform various preprocessing steps on a time-series dataset.
    """
    def __init__(self, filename=None, dataframe=None):
        """
        Initializes the Preprocessing class. Can take a filename to load or an existing dataframe.
        """
        self.filename = filename
        self.dataframe = dataframe

    def load_dataframe(self):
        """
        Loads a CSV file into a Pandas DataFrame and converts the 'timestamp' column to datetime.
        Handles both file paths and uploaded file objects from Streamlit.
        """
        data_raw_df = None
        try:
            if self.filename:
                data_raw_df = pd.DataFrame(pd.read_csv(self.filename))
            elif self.dataframe is not None:
                # If a dataframe is directly provided (e.g., from st.file_uploader)
                data_raw_df = pd.DataFrame(self.dataframe)
            else:
                st.error("No filename or dataframe provided for loading.")
                return None

            if 'timestamp' in data_raw_df.columns:
                data_raw_df['timestamp'] = pd.to_datetime(data_raw_df['timestamp'], format='mixed', errors='coerce')
                # Drop rows where timestamp conversion failed
                data_raw_df.dropna(subset=['timestamp'], inplace=True)
            else:
                st.warning("No 'timestamp' column found. Proceeding without datetime conversion.")
                
        except Exception as e:
            st.error(f"Unable to open or process the file: {e}")
            return None
        return data_raw_df
    
    def remove_outliers(self, df_input):
        """
        Removes outliers from numeric columns using the IQR method.
        Outliers are defined as values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
        """
        df = df_input.copy()

        numeric_cols = []
        for col in df.columns:
            if col != 'timestamp': # Exclude timestamp from outlier detection
                df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce non-numeric to NaN
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)

        if not numeric_cols:
            st.warning("No numeric columns found for outlier detection. Returning original DataFrame.")
            return df # Return df as is if no numeric columns

        # Initialize a mask that assumes all rows are valid
        combined_mask = pd.Series(True, index=df.index, dtype=bool)

        for col in numeric_cols:
            col_series_valid = df[col].dropna() # Only consider non-NaN values for IQR calculation
            if col_series_valid.empty:
                st.info(f"Column '{col}' is empty after dropping NaNs, skipping outlier detection for it.")
                continue

            # Calculate IQR bounds
            q1 = col_series_valid.quantile(0.25)
            q3 = col_series_valid.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Create a mask for the current column's outliers
            # Include NaN values in the mask to be handled later (they won't be removed by this filter)
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            # Combine with previous masks: a row is kept only if it's not an outlier in ANY numeric column
            combined_mask = combined_mask & col_mask
        
        # Apply the combined mask to the original input DataFrame to preserve original dtypes if possible
        filtered_df = df_input[combined_mask]
        
        st.info(f"Removed {len(df_input) - len(filtered_df)} outliers.")
        return filtered_df
        
    # Remove the old scaling_data method. We will scale after splitting.
    # def scaling_data(self, df_input):
    #     """
    #     Scales numeric columns of the DataFrame using MinMaxScaler.
    #     Returns the scaled DataFrame and the scaler object.
    #     """
    #     # ... (old code removed)

    def drop_irrelevant_cols(self, df):
        """
        Drops predefined irrelevant columns from the DataFrame.
        """
        drop_columns = ['servers_power_kw_internal','storage_total_power_kw','network devices total power (Kw)']
        # Filter out columns that don't exist in the current DataFrame
        existing_drop_columns = [col for col in drop_columns if col in df.columns]
        
        if existing_drop_columns:
            df_relevant_cols = df.drop(columns=existing_drop_columns, axis=1)
            st.info(f"Dropped irrelevant columns: {', '.join(existing_drop_columns)}")
        else:
            df_relevant_cols = df.copy()
            st.info("No predefined irrelevant columns found to drop.")
        return df_relevant_cols
    
    def add_new_cols(self, df):
        """
        Adds 'total_power_consumption' and 'PUE' columns to the DataFrame.
        """
        if df.empty:
            st.warning("DataFrame is empty, cannot add new columns.")
            return df

        required_cols = [
            'IT power consumption', 'UPS_total_power(Kw)', 'PDU_total_power(Kw)',
            'lights_total_power(Kw)', 'cooling_power_kw_internal'
        ]
        
        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"Missing columns for 'total_power_consumption' calculation: {', '.join(missing_cols)}. Skipping addition of new columns.")
            return df.copy() # Return a copy to avoid modifying original df if not adding columns

        df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
        df_copy['total_power_consumption'] = (
            df_copy['IT power consumption'] + df_copy['UPS_total_power(Kw)'] +
            df_copy['PDU_total_power(Kw)'] + df_copy['lights_total_power(Kw)'] +
            df_copy['cooling_power_kw_internal']
        )
        
        # Avoid division by zero for PUE
        df_copy['PUE'] = df_copy.apply(
            lambda row: row['total_power_consumption'] / row['IT power consumption'] 
            if row['IT power consumption'] != 0 else np.nan, axis=1
        )
        st.success("Added 'total_power_consumption' and 'PUE' columns.")
        return df_copy
    
    def data_split(self, raw_data, train_data_percent=.8):
        """
        Splits the raw data into training and testing sets based on a percentage.
        """
        if raw_data.empty:
            st.warning("Input data is empty, cannot split.")
            return pd.DataFrame(), pd.DataFrame()

        training_data_len = math.ceil(len(raw_data) * train_data_percent)
        train_data = raw_data.iloc[:training_data_len, :] 
        test_data = raw_data.iloc[training_data_len:, :]
        st.success(f"Data split: Training set size = {len(train_data)}, Test set size = {len(test_data)}")
        return train_data , test_data

    def create_dataset(self, dataset, look_back_window, target_column_index):
        """
        Creates X (features) and Y (target) datasets for time series forecasting.
        X contains sequences of 'look_back_window' observations.
        Y contains the target value for the next step from the specified target_column_index.
        
        Note: This function now expects already scaled data for `dataset`.
        It returns X with all features and Y as the single target feature.
        """
        dataX , dataY = [],[]
        
        # Ensure dataset is a NumPy array for consistent indexing
        if isinstance(dataset, pd.DataFrame):
            dataset_np = dataset.values
        else:
            dataset_np = dataset

        if len(dataset_np) <= look_back_window:
            st.warning(f"Dataset length ({len(dataset_np)}) is too short for look_back_window ({look_back_window}). Cannot create dataset.")
            return np.array([]), np.array([])

        # Check if the target_column_index is valid
        if not (-dataset_np.shape[1] <= target_column_index < dataset_np.shape[1]):
            st.error(f"Invalid target_column_index: {target_column_index}. Dataset has {dataset_np.shape[1]} columns.")
            return np.array([]), np.array([])
            
        for i in range(len(dataset_np) - look_back_window):
            # Features: 'look_back_window' rows, all columns from the input dataset
            a = dataset_np[i:(i + look_back_window), :]
            dataX.append(a)
            # Target: The value from the specified target_column_index at the next step
            dataY.append(dataset_np[i + look_back_window, target_column_index])
            
        # Reshape dataY to be a 2D array (number_of_samples, 1)
        return np.array(dataX), np.reshape(np.array(dataY), (np.array(dataY).shape[0], 1))
        
    def input_output_reshape(self, X_data, Y_data):
        """
        Reshapes input and output arrays for LSTM model compatibility.
        X_data: (samples, timesteps, features)
        Y_data: (samples, 1)
        """
        if X_data.size == 0 or Y_data.size == 0:
            st.warning("Input arrays are empty, cannot reshape.")
            return np.array([]), np.array([])

        X_array , Y_array = np.array(X_data) , np.array(Y_data)

        # X_array is already (num_samples, look_back_window, num_features) from create_dataset
        # So, it's already in the correct format for a Keras LSTM layer that accepts multivariate input.
        X_array_reshape = X_array 

        Y_array_reshape = np.reshape(Y_array,(Y_array.shape[0],1))
        st.success("Input and output arrays reshaped successfully.")
        return X_array_reshape , Y_array_reshape


# ---
# Streamlit App Layout
# ---
st.set_page_config(layout="wide", page_title="Data Preprocessing App",page_icon="⚙️",)

st.title("⚙️ Data Preprocessing for Time Series")
st.markdown("""
This application helps you preprocess your time-series data for machine learning models.
Upload your CSV file and follow the steps.
""")

# Initialize variables for Streamlit's state
df_raw = None
df_raw_no_outliers = None
df_relevant_cols = None
df_new_cols = None
train_df_unscaled = None # Unscaled train data (features + target)
test_df_unscaled = None  # Unscaled test data (features + target)
scaler_X = None          # Scaler for features (X)
scaler_y = None          # Scaler for target (y)
X_train_scaled = None
y_train_scaled = None
X_test_scaled = None
y_test_scaled = None
train_X_reshaped = None
train_y_reshaped = None
test_X_reshaped = None
test_y_reshaped = None

# File Uploader
st.header("1. Upload Your CSV Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Initialize Preprocessing class with the uploaded file
    preprocess_obj = Preprocessing(dataframe=pd.read_csv(uploaded_file))

    st.success("File uploaded successfully!")

    with st.spinner("Loading and preparing data..."):
        df_raw = preprocess_obj.load_dataframe()
    
    if df_raw is not None and not df_raw.empty:
        st.subheader("Original Data (First 5 rows)")
        st.dataframe(df_raw.head())

        # 2. Remove Outliers
        st.header("2. Outlier Removal (IQR Method)")
        with st.spinner("Removing outliers..."):
            df_raw_no_outliers = preprocess_obj.remove_outliers(df_raw)
        
        if df_raw_no_outliers is not None and not df_raw_no_outliers.empty:
            st.subheader("Data After Outlier Removal (First 5 rows)")
            st.dataframe(df_raw_no_outliers.head())
            st.info(f"Original rows: {len(df_raw)}, Rows after outlier removal: {len(df_raw_no_outliers)}")
        else:
            st.warning("No data remaining after outlier removal or an issue occurred.")

        # 3. Drop Irrelevant Columns
        st.header("3. Drop Irrelevant Columns")
        if df_raw_no_outliers is not None:
            with st.spinner("Dropping columns..."):
                df_relevant_cols = preprocess_obj.drop_irrelevant_cols(df_raw_no_outliers)
            
            if df_relevant_cols is not None and not df_relevant_cols.empty:
                st.subheader("Data After Dropping Columns (First 5 rows)")
                st.dataframe(df_relevant_cols.head())
                st.info(f"Columns remaining: {df_relevant_cols.shape[1]}")
            else:
                st.warning("No data or columns remaining after dropping irrelevant columns.")

        # 4. Add New Columns
        st.header("4. Add New Features")
        if df_relevant_cols is not None:
            with st.spinner("Adding new columns (total_power_consumption, PUE)..."):
                df_new_cols = preprocess_obj.add_new_cols(df_relevant_cols)
            
            if df_new_cols is not None and not df_new_cols.empty:
                st.subheader("Data After Adding New Columns (First 5 rows)")
                st.dataframe(df_new_cols.head())
                st.info(f"New columns added. Total columns: {df_new_cols.shape[1]}")
            else:
                st.warning("No data or an issue occurred while adding new columns.")
        
        # --- NEW LOGIC: Split first, then Scale ---
        
        # 5. Data Split (on unscaled data)
        st.header("5. Data Split (Train/Test - **Unscaled**)")
        if df_new_cols is not None:
            train_percent = st.slider("Select Training Data Percentage", 0.0, 1.0, 0.8, 0.05, key="train_split_slider")
            with st.spinner("Splitting data..."):
                train_df_unscaled, test_df_unscaled = preprocess_obj.data_split(df_new_cols, train_data_percent=train_percent)
            
            if train_df_unscaled is not None and not train_df_unscaled.empty:
                st.subheader("Unscaled Training Data (First 5 rows)")
                st.dataframe(train_df_unscaled.head())
            if test_df_unscaled is not None and not test_df_unscaled.empty:
                st.subheader("Unscaled Test Data (First 5 rows)")
                st.dataframe(test_df_unscaled.head())
            else:
                st.warning("No data or an issue occurred while splitting data.")

        # 6. Separate X and Y, then Scale X and Y with different scalers
        st.header("6. Scale Features (X) and Target (Y) Separately")
        if train_df_unscaled is not None and test_df_unscaled is not None:
            
            # Get column names for dynamic selection
            available_columns = [col for col in df_new_cols.columns.tolist() if pd.api.types.is_numeric_dtype(df_new_cols[col])]
            
            default_target_col = None
            if 'IT power consumption' in available_columns:
                default_target_col = 'IT power consumption'
            elif available_columns:
                default_target_col = available_columns[-1]

            if not available_columns:
                st.error("No numeric columns available to select as target. Please check your data.")
            else:
                selected_target_column_name = st.selectbox(
                    "Select Target Column (Y) for Scaling", 
                    options=available_columns, 
                    index=available_columns.index(default_target_col) if default_target_col else 0,
                    key="target_col_select_scaling"
                )
                
                # Separate X and Y for both train and test sets
                X_train_unscaled = train_df_unscaled.drop(columns=[selected_target_column_name, 'timestamp'], errors='ignore')
                y_train_unscaled = train_df_unscaled[[selected_target_column_name]] # Keep as DataFrame for scaler

                X_test_unscaled = test_df_unscaled.drop(columns=[selected_target_column_name, 'timestamp'], errors='ignore')
                y_test_unscaled = test_df_unscaled[[selected_target_column_name]] # Keep as DataFrame for scaler

                st.info(f"Selected '{selected_target_column_name}' as the target column. Features for X will be all other numeric columns.")

                with st.spinner("Fitting and transforming X and Y with separate MinMaxScaler objects..."):
                    # Initialize separate scalers
                    scaler_X = MinMaxScaler()
                    scaler_y = MinMaxScaler()

                    # Fit and transform training data
                    X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train_unscaled), columns=X_train_unscaled.columns, index=X_train_unscaled.index)
                    y_train_scaled = pd.DataFrame(scaler_y.fit_transform(y_train_unscaled), columns=y_train_unscaled.columns, index=y_train_unscaled.index)

                    # Transform test data (DO NOT FIT AGAIN)
                    X_test_scaled = pd.DataFrame(scaler_X.transform(X_test_unscaled), columns=X_test_unscaled.columns, index=X_test_unscaled.index)
                    y_test_scaled = pd.DataFrame(scaler_y.transform(y_test_unscaled), columns=y_test_unscaled.columns, index=y_test_unscaled.index)
                    
                    st.success("Features (X) and Target (Y) scaled successfully!")
                    st.write(f"**Shape of X_train_scaled:** {X_train_scaled.shape}")
                    st.write(f"**Shape of y_train_scaled:** {y_train_scaled.shape}")
                    st.write(f"**Shape of X_test_scaled:** {X_test_scaled.shape}")
                    st.write(f"**Shape of y_test_scaled:** {y_test_scaled.shape}")

                    st.subheader("Scaled X_train (First 5 rows)")
                    st.dataframe(X_train_scaled.head())
                    st.subheader("Scaled y_train (First 5 rows)")
                    st.dataframe(y_train_scaled.head())

                    # Save the scalers
                    scaler_dir = "scaler_data"
                    os.makedirs(scaler_dir, exist_ok=True)
                    joblib.dump(scaler_X, os.path.join(scaler_dir, "scaler_X.joblib"))
                    joblib.dump(scaler_y, os.path.join(scaler_dir, "scaler_y.joblib"))
                    st.info(f"Scalers saved to '{scaler_dir}/scaler_X.joblib' and '{scaler_dir}/scaler_y.joblib'")

        # 7. Create Dataset for LSTM (using the scaled data)
        st.header("7. Create Time Series Dataset (X, Y) for Model Training")
        if X_train_scaled is not None and y_train_scaled is not None and X_test_scaled is not None and y_test_scaled is not None:
            look_back = st.slider("Select Look Back Window (Timesteps)", 1, 100, 48, key="look_back_slider")
            
            # Combine X and Y scaled data for create_dataset, ensuring target is last for easier indexing
            # The 'create_dataset' function expects a single dataset with features and target
            # It's crucial that the target_column_index passed to create_dataset corresponds to the index
            # of the target within the combined array.
            
            # First, make sure columns are aligned. Let's create a combined DataFrame for 'create_dataset'
            # The column order matters for target_column_index
            
            # Get the actual columns order after droping timestamp and target from X
            X_cols = X_train_scaled.columns.tolist()
            y_col = y_train_scaled.columns.tolist()[0] # Get the single column name of y

            # Create a full dataset (scaled) to pass to create_dataset
            # Ensure the target column is placed correctly at the end or its index is known.
            # Here, we'll append the target as the last column for simplicity in create_dataset.
            train_combined_scaled = pd.concat([X_train_scaled, y_train_scaled], axis=1)
            test_combined_scaled = pd.concat([X_test_scaled, y_test_scaled], axis=1)
            
            # The target column will now be the very last column in the combined DataFrame
            target_column_index_in_combined = train_combined_scaled.shape[1] - 1 
            
            st.info(f"Target column '{y_col}' is at index {target_column_index_in_combined} in the combined feature+target dataset.")

            with st.spinner("Creating training dataset (X_train, y_train)..."):
                train_X, train_y = preprocess_obj.create_dataset(train_combined_scaled.values, look_back_window=look_back, target_column_index=target_column_index_in_combined)
            
            with st.spinner("Creating test dataset (X_test, y_test)..."):
                test_X, test_y = preprocess_obj.create_dataset(test_combined_scaled.values, look_back_window=look_back, target_column_index=target_column_index_in_combined)
            
            if train_X.size > 0 and train_y.size > 0:
                st.success("Datasets created successfully!")
                st.write(f"**Shape of X_train (LSTM input):** {train_X.shape}")
                st.write(f"**Shape of y_train (LSTM target):** {train_y.shape}")
                st.write(f"**Shape of X_test (LSTM input):** {test_X.shape}")
                st.write(f"**Shape of y_test (LSTM target):** {test_y.shape}")

                # 8. Reshape for LSTM (if needed, based on model architecture)
                st.header("8. Reshape for Model Input")
                st.info("This step ensures X is (samples, timesteps, features) and Y is (samples, 1).")
                with st.spinner("Reshaping X and Y for Model..."):
                    train_X_reshaped, train_y_reshaped = preprocess_obj.input_output_reshape(train_X, train_y)
                    test_X_reshaped, test_y_reshaped = preprocess_obj.input_output_reshape(test_X, test_y)

                if train_X_reshaped.size > 0 and train_y_reshaped.size > 0:
                    st.success("Reshaping complete!")
                    st.write(f"**Reshaped X_train:** {train_X_reshaped.shape}")
                    st.write(f"**Reshaped y_train:** {train_y_reshaped.shape}")
                    st.write(f"**Reshaped X_test:** {test_X_reshaped.shape}")
                    st.write(f"**Reshaped y_y_test:** {test_y_reshaped.shape}")

                    # --- Save the reshaped data for the training app ---
                    st.subheader("Save Preprocessed Data")
                    if st.button("Save Data for Model Training"):
                        data_save_dir = "preprocessed_data" # New directory for saving npy files
                        os.makedirs(data_save_dir, exist_ok=True)
                        try:
                            np.save(os.path.join(data_save_dir, 'train_X.npy'), train_X_reshaped)
                            np.save(os.path.join(data_save_dir, 'train_y.npy'), train_y_reshaped)
                            np.save(os.path.join(data_save_dir, 'test_X.npy'), test_X_reshaped)
                            np.save(os.path.join(data_save_dir, 'test_y.npy'), test_y_reshaped)
                            st.success(f"Preprocessed data saved as .npy files in '{data_save_dir}/'.")
                            st.info("You can now upload these files to the ' Model Training & Evaluation' app.")
                        except Exception as e:
                            st.error(f"Error saving data: {e}")
                else:
                    st.warning("Reshaping resulted in empty arrays or an error.")
            else:
                st.warning("Dataset creation resulted in empty arrays or an error.")
        else:
            st.info("Please complete previous steps to create time series datasets.")

else:
    st.info("Please upload a CSV file to begin preprocessing.")