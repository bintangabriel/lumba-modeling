from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from pandas.core.frame import DataFrame

from typing import Any, Optional, Union, List

class LumbaLinearRegression:
    model: LinearRegression

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe = dataframe

    def train_model(self, train_column_name: Union[str, List[str]], target_column_name: str, train_size: float = 0.8) -> dict:
        if self.dataframe[target_column_name].dtype not in ["int64", "float64"]:
            return {
                'error': 'Kolom yang boleh dipilih hanyalah kolom dengan data numerik saja. Silakan pilih kolom yang benar.'
            }
        
        x = None
        if type(train_column_name) == str:
            if self.dataframe[train_column_name].dtype not in ["int64", "float64"]:
                return {
                    'error': 'Kolom yang boleh dipilih hanyalah kolom dengan data numerik saja. Silakan pilih kolom yang benar.'
                }
            x = self.dataframe[train_column_name].to_numpy().reshape(-1, 1)
        
        elif type(train_column_name) == list:
            for col in train_column_name:
                if self.dataframe[col].dtype not in ["int64", "float64"]:
                    return {
                        'error': 'Kolom yang boleh dipilih hanyalah kolom dengan data numerik saja. Silakan pilih kolom yang benar.'
                    }

            x = self.dataframe[train_column_name].to_numpy()

        y = self.dataframe[target_column_name].to_numpy().reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
        
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        
        y_pred = lr.predict(x_test)
        r2 = r2_score(y_true=y_test, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        mse = mean_squared_error(y_true=y_test, y_pred=y_pred)

        self.model = lr

        return {
            'model': lr,
            'r2_score': f'{r2:.4f}',
            'mean_absolute_error': f'{mae:.4f}',
            'mean_squared_error': f'{mse:.4f}',
        }

    def get_model(self) -> Optional[LinearRegression]:
        try:
            return self.model
        except AttributeError:
            return None

    def predict(self, data_target: Any) -> Any:
        lr = self.get_model()
        y_pred = lr.predict(data_target)

        return y_pred