from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from pandas.core.frame import DataFrame

from typing import Any

class LumbaLinearRegression:
    model: LinearRegression = None

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe = dataframe

    def train_model(self, train_column_name: str, target_column_name:str, train_size: float = 0.8) -> dict:
        x = self.dataframe[train_column_name].to_numpy().reshape(-1, 1)
        y = self.dataframe[target_column_name].to_numpy().reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
        
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        
        y_pred = lr.predict(x_test)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)

        self.model = lr

        return {
            'model': lr,
            'mean_absolute_error': f'{mae:.4f}'
        }

    def get_model(self) -> LinearRegression:
        return self.model

    def predict(self, data_target: Any) -> Any:
        lr = self.get_model()
        y_pred = lr.predict(data_target)

        return y_pred