from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout, Input
from bayes_opt import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import xgboost as xgb # Import XGBoost
from sklearn.metrics import mean_squared_error # For XGBoost evaluation

class ModelTrainTestOptimize:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train_final = X_train
        self.Y_train_final = y_train
        self.X_test_final = X_test
        self.Y_test_final = y_test
        
        # Determine input_shape for Keras models (RNN, LSTM, GRU)
        # Assuming X_train is 3D: (samples, timesteps, features)
        if X_train.ndim == 3:
            look_back_window = X_train.shape[1]
            num_features_in_X = X_train.shape[2]
            self.input_shape = (look_back_window, num_features_in_X)
        else:
            # Handle cases where X_train might already be 2D (e.g., if pre-processed for non-RNN models)
            # For this context, we primarily expect 3D for RNNs.
            # If X_train is 2D (samples, features), input_shape would be (features,)
            self.input_shape = (X_train.shape[1],) 
            print("Warning: X_train is not 3D. Ensure input_shape is correctly handled for Keras models.")

        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

    def _reshape_for_xgboost(self, X_data):
        """
        Reshapes 3D (samples, timesteps, features) to 2D (samples, timesteps * features) for XGBoost.
        If data is already 2D, returns as is.
        """
        if X_data.ndim == 3:
            samples, timesteps, features = X_data.shape
            return X_data.reshape(samples, timesteps * features)
        return X_data # Already 2D or other shape, return as is

    def _build_rnn_model(self, num_layers, units, dropout_rate, learning_rate):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        
        for i in range(int(num_layers)):
            # return_sequences=True for all layers except the last one
            return_sequences = (i < int(num_layers) - 1)
            model.add(SimpleRNN(units=int(units), return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
            
        model.add(Dense(units=1, activation="linear"))
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss="mse", optimizer=optimizer, metrics=['mse'])
        
        return model

    def rnn_objective(self, num_layers, units, dropout_rate, learning_rate):
        try:
            model = self._build_rnn_model(num_layers, units, dropout_rate, learning_rate)
            history = model.fit(self.X_train_final, self.Y_train_final, epochs=100, batch_size=32, callbacks=[self.early_stopping], verbose=0, validation_data=(self.X_test_final, self.Y_test_final)) 
            mse = history.history['val_mse'][-1]
            return -mse
        except Exception as e:
            print(f"An error occurred during RNN training: {e}")
            return -np.inf

    def _build_lstm_model(self, num_layers, units, dropout_rate, learning_rate):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        
        for i in range(int(num_layers)):
            return_sequences = (i < int(num_layers) - 1)
            model.add(LSTM(units=int(units), return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(units=1, activation="linear"))
    
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss="mse", optimizer=optimizer, metrics=['mse'])
    
        return model

    def lstm_objective(self, num_layers, units, dropout_rate, learning_rate):
        try:
            model = self._build_lstm_model(num_layers, units, dropout_rate, learning_rate)
            history = model.fit(self.X_train_final, self.Y_train_final, epochs=100, batch_size=32, verbose=0, callbacks=[self.early_stopping], validation_data=(self.X_test_final, self.Y_test_final)) 
            mse = history.history['val_mse'][-1]
            return -mse
        except Exception as e:
            print(f"An error occurred during LSTM training: {e}")
            return -np.inf

    def _build_gru_model(self, num_layers, units, dropout_rate, learning_rate):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        
        for i in range(int(num_layers)):
            return_sequences = (i < int(num_layers) - 1)
            model.add(GRU(units=int(units), return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(units=1, activation="linear"))
    
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss="mse", optimizer=optimizer, metrics=['mse'])
    
        return model

    def gru_objective(self, num_layers, units, dropout_rate, learning_rate):
        try:
            model = self._build_gru_model(num_layers, units, dropout_rate, learning_rate)
            history = model.fit(self.X_train_final, self.Y_train_final, epochs=100, batch_size=32, callbacks=[self.early_stopping], verbose=0, validation_data=(self.X_test_final, self.Y_test_final)) 
            mse = history.history['val_mse'][-1]
            return -mse
        except Exception as e:
            print(f"An error occurred during GRU training: {e}")
            return -np.inf

    def xgboost_objective(self, n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
        """
        Objective function for Bayesian Optimization of XGBoost.
        """
        # Reshape training data for XGBoost
        train_X_flat = self._reshape_for_xgboost(self.X_train_final)
        
        # Initialize and train XGBoost Regressor
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1
        )
        # Flatten Y_train_final for XGBoost if it's 2D (samples, 1)
        model.fit(train_X_flat, self.Y_train_final.flatten()) 

        # Evaluate on test set
        test_X_flat = self._reshape_for_xgboost(self.X_test_final)
        # Flatten Y_test_final for MSE calculation if it's 2D (samples, 1)
        y_pred = model.predict(test_X_flat)
        mse = mean_squared_error(self.Y_test_final.flatten(), y_pred.flatten())
        
        return -mse # Bayesian Optimization maximizes, so return negative MSE

    def model_testing(self, model, scaled, model_type):
        """
        Makes predictions and inverse transforms them.
        Handles different input shapes for Keras and XGBoost models.
        """
        if model_type in ['rnn', 'lstm', 'gru']:
            Y_pred_scaled = model.predict(self.X_test_final)
        elif model_type == 'xgboost':
            X_test_flat = self._reshape_for_xgboost(self.X_test_final)
            Y_pred_scaled = model.predict(X_test_flat)
            # XGBoost predicts 1D array, reshape to 2D (samples, 1) for inverse_transform if necessary
            Y_pred_scaled = Y_pred_scaled.reshape(-1, 1)
        else:
            raise ValueError(f"Invalid model_type for model_testing: {model_type}")

        # Ensure Y_test_final is 2D (samples, 1) for inverse_transform
        Y_test_final_2d = self.Y_test_final.reshape(-1, 1)

        y_pred_retransform = scaled.inverse_transform(Y_pred_scaled)
        y_test_retransform = scaled.inverse_transform(Y_test_final_2d)
        return y_pred_retransform, y_test_retransform

    def model_fitting_testing(self, optimizer, model_name, scale_obj):
        # Extract best parameters, handling both RNN/LSTM/GRU and XGBoost structures
        best_params = optimizer.max['params']
        
        model = None
        if model_name in ['rnn', 'lstm', 'gru']:
            best_num_layers = int(best_params['num_layers'])
            best_units = int(best_params['units'])
            best_dropout_rate = best_params['dropout_rate']
            best_learning_rate = best_params['learning_rate']

            if model_name == 'rnn':
                model = self._build_rnn_model(best_num_layers, best_units, best_dropout_rate, best_learning_rate)
            elif model_name == 'lstm':
                model = self._build_lstm_model(best_num_layers, best_units, best_dropout_rate, best_learning_rate)
            elif model_name == 'gru':
                model = self._build_gru_model(best_num_layers, best_units, best_dropout_rate, best_learning_rate)
            
            # Fit Keras models
            model.fit(self.X_train_final, self.Y_train_final, epochs=50, batch_size=32, validation_data=(self.X_test_final, self.Y_test_final), verbose=0)
            
        elif model_name == 'xgboost':
            best_n_estimators = int(best_params['n_estimators'])
            best_max_depth = int(best_params['max_depth'])
            best_learning_rate = best_params['learning_rate']
            best_subsample = best_params['subsample']
            best_colsample_bytree = best_params['colsample_bytree']

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=best_n_estimators,
                max_depth=best_max_depth,
                learning_rate=best_learning_rate,
                subsample=best_subsample,
                colsample_bytree=best_colsample_bytree,
                random_state=42,
                n_jobs=-1
            )
            # Reshape training data for final fit
            train_X_flat = self._reshape_for_xgboost(self.X_train_final)
            model.fit(train_X_flat, self.Y_train_final.flatten()) # Flatten Y_train_final for XGBoost
        else:
            raise ValueError("Invalid model_name provided. Choose 'rnn', 'lstm', 'gru', or 'xgboost'.")
        
        # Make predictions and inverse transform
        y_pred, y_test = self.model_testing(model, scale_obj, model_name)
        
        return y_pred, y_test

class ModelSelection:
    def __init__(self, obj_ModelTrainTestOptimize): # Renamed 'object' to 'obj_ModelTrainTestOptimize' for clarity
        self.obj = obj_ModelTrainTestOptimize

    def final_model_selection(self, optimizer, mdl_nm):
        print(f"Model Selected is : {mdl_nm}")
        best_params = optimizer.max['params']
        
        model = None
        if mdl_nm in ['rnn', 'lstm', 'gru']:
            best_num_layers = int(best_params['num_layers'])
            best_units = int(best_params['units'])
            best_dropout_rate = best_params['dropout_rate']
            best_learning_rate = best_params['learning_rate']

            if mdl_nm == 'rnn':
                model = self.obj._build_rnn_model(best_num_layers, best_units, best_dropout_rate, best_learning_rate)
            elif mdl_nm == 'lstm':
                model = self.obj._build_lstm_model(best_num_layers, best_units, best_dropout_rate, best_learning_rate)
            elif mdl_nm == 'gru':
                model = self.obj._build_gru_model(best_num_layers, best_units, best_dropout_rate, best_learning_rate)
        elif mdl_nm == 'xgboost':
            best_n_estimators = int(best_params['n_estimators'])
            best_max_depth = int(best_params['max_depth'])
            best_learning_rate = best_params['learning_rate']
            best_subsample = best_params['subsample']
            best_colsample_bytree = best_params['colsample_bytree']

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=best_n_estimators,
                max_depth=best_max_depth,
                learning_rate=best_learning_rate,
                subsample=best_subsample,
                colsample_bytree=best_colsample_bytree,
                random_state=42,
                n_jobs=-1
            )
            # Fit the final model on the entire training data
            train_X_flat = self.obj._reshape_for_xgboost(self.obj.X_train_final)
            model.fit(train_X_flat, self.obj.Y_train_final.flatten())
        else:
            raise ValueError("Invalid model name. Choose 'rnn', 'lstm', 'gru', or 'xgboost'.")

        return model, optimizer
