# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, r2_score

# from pandas.core.frame import DataFrame
# import numpy as np
# from typing import Optional, Any, Tuple

# class LumbaLSTM:
#     model: Sequential
#     scaler: MinMaxScaler
#     train_column_name: str
#     n_steps: int

#     def __init__(self, dataframe: DataFrame) -> None:
#         self.dataframe = dataframe

#     def train_model(self, train_column_name: str, steps: int = 4) -> dict:
#         if self.dataframe[train_column_name].dtype not in ["int64", "float64"]:
#             return {
#                 'error': 'Kolom yang boleh dipilih hanyalah kolom dengan data numerik saja. Silakan pilih kolom yang benar.'
#             }
        
#         scaler = MinMaxScaler()
#         scaled_data = scaler.fit_transform(self.dataframe[train_column_name].values.reshape(-1, 1))

#         # prepare the training and testing data
#         n_steps = steps
#         train_data = scaled_data[:-10]
#         test_data = scaled_data[-10:]
#         X_train = []
#         y_train = []
#         for i in range(n_steps, len(train_data)):
#             X_train.append(train_data[i-n_steps:i, 0])
#             y_train.append(train_data[i, 0])
#         X_train, y_train = np.array(X_train), np.array(y_train)
#         X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#         X_test = []
#         y_test = self.dataframe[train_column_name][-10:]
#         scaled_test_data = scaler.transform(y_test.values.reshape(-1, 1))
#         for i in range(n_steps, len(scaled_test_data)):
#             X_test.append(scaled_test_data[i-n_steps:i, 0])
#         X_test = np.array(X_test)
#         X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#         # define the LSTM model
#         lstm_model = Sequential()
#         lstm_model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
#         lstm_model.add(Dense(1))
#         lstm_model.compile(optimizer='adam', loss='mse')

#         # train the LSTM model on the training data
#         lstm_model.fit(X_train, y_train, epochs=100, verbose=0)

#         # use the trained LSTM model to make forecasts for the next 10 weeks
#         y_pred_test = lstm_model.predict(X_test)
#         y_pred_test = scaler.inverse_transform(y_pred_test).ravel()
#         y_test = self.dataframe[train_column_name][-6:]

#         # calculate the RMSE, MSE, and R2 Score for the test data
#         lstm_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
#         lstm_mse = mean_squared_error(y_test, y_pred_test)
#         lstm_r2 = r2_score(y_test, y_pred_test)

#         self.model = lstm_model
#         self.scaler = scaler
#         self.train_column_name = train_column_name
#         self.n_steps = n_steps
#         self.scaled_test_data = scaled_test_data
        
#         return {
#             'model': lstm_model,
#             'scaled_test_data': scaled_test_data,
#             'r2_score': f'{lstm_r2:.4f}',
#             'scaler': scaler,
#             'mean_squared_error': f'{lstm_mse:.4f}',
#             'root_mean_squared_error': f'{lstm_rmse:.4f}',
#         }
    
#     def get_model(self) -> Optional[Sequential]:
#         try:
#             return self.model
#         except AttributeError:
#             return None
        
#     def predict(self, n_week: int) -> Tuple[Any, Any, Any]:
#         """Forecast for the n weeks ahead"""

#         forecast_data = self.dataframe[self.train_column_name].values.tolist()
#         lstm_model = self.get_model()
#         for _ in range(n_week):
#             forecast_input = self.scaled_test_data[-self.n_steps:]
#             forecast_input = forecast_input.reshape((1, self.n_steps, 1))
#             forecast = lstm_model.predict(forecast_input)
#             forecast_data.append(self.scaler.inverse_transform(forecast)[0][0])
#             self.scaled_test_data = np.append(self.scaled_test_data, forecast)

#         # calculate the upper and lower bounds for the forecasted sales data
#         upper_bound = np.array(forecast_data) * 1.1
#         lower_bound = np.array(forecast_data) * 0.9

#         return forecast_data, upper_bound, lower_bound