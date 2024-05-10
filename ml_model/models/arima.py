from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, Any, Optional
from sklearn.metrics import mean_squared_error, r2_score
from pandas.core.frame import DataFrame
import pandas as pd
import itertools
import numpy as np


class LumbaARIMA:
    model: ARIMA
    train_column_name: str

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe = dataframe

    def train_model(self, train_column_name: str, split_size: float = 0.8) -> dict:
        if self.dataframe[train_column_name].dtype not in ["int64", "float64"]:
            return {
                'error': 'Kolom yang boleh dipilih hanyalah kolom dengan data numerik saja. Silakan pilih kolom yang benar.'
            }
        
        # split the data into training and testing sets
        train_size = int(len(self.dataframe) * split_size)
        train_data, test_data = self.dataframe[:train_size], self.dataframe[train_size:]

        best_pdq = self._find_best_pdq(train_data, train_column_name)

        # fit an ARIMA model to the training data
        arima_model = ARIMA(train_data[train_column_name], order=best_pdq)
        arima_model_fit = arima_model.fit()

        # make predictions on the testing set
        y_pred = arima_model_fit.forecast(steps=len(test_data))

        # evaluate the accuracy of the ARIMA model using RMSE
        arima_mse = mean_squared_error(test_data[train_column_name], y_pred)
        arima_rmse = np.sqrt(arima_mse)
        arima_r2_score = r2_score(test_data[train_column_name], y_pred)

        self.model = arima_model_fit
        self.train_column_name = train_column_name

        return {
            'model': arima_model_fit,
            'r2_score': f'{arima_r2_score:.4f}',
            'mean_squared_error': f'{arima_mse:.4f}',
            'root_mean_squared_error': f'{arima_rmse:.4f}',
        }
        
    def get_model(self) -> Optional[ARIMA]:
        try:
            return self.model
        except AttributeError:
            return None

    def predict(self, n_week: int, week_column_name: str) -> Tuple[Any, Any, Any]:
        """Forecast for the n weeks ahead"""

        # use the trained ARIMA model to make forecasts for the next 10 weeks
        arima_forecast_data = self.dataframe.copy()
        arima_forecast_data[week_column_name] = pd.to_numeric(arima_forecast_data[week_column_name], downcast="integer")
        arima_forecast_data[self.train_column_name] = pd.to_numeric(arima_forecast_data[self.train_column_name], downcast="integer")
        arima_forecast = self.model.forecast(steps=n_week)

        for i, forecast_value in enumerate(arima_forecast):
            week_num = len(self.dataframe) + i + 1
            new = [week_num, forecast_value]
            arima_forecast_data = arima_forecast_data.append(pd.Series(new, index=arima_forecast_data.columns[:len(new)]), ignore_index=True)

        # calculate the upper and lower bounds for the forecasted sales data
        upper_bound = np.array(arima_forecast_data) * 1.1
        lower_bound = np.array(arima_forecast_data) * 0.9

        return arima_forecast_data, upper_bound, lower_bound

    @staticmethod
    def _find_best_pdq(train_data, train_column_name: str) -> Tuple[Any, Any, Any]:
        # Define the range of values to search over
        p_range = range(0, 3)
        d_range = range(0, 3)
        q_range = range(0, 3)
        pdq = list(itertools.product(p_range, d_range, q_range))

        # Select the model with the lowest AIC
        lowest_aic = float('inf')
        best_pdq = None
        for param in pdq:
            try:
                model = ARIMA(train_data[train_column_name], order=param)
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < lowest_aic:
                    lowest_aic = aic
                    best_pdq = param
            except:
                continue

        return best_pdq  # (p, d, q)
