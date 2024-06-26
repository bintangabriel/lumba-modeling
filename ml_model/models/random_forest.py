from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from pandas.core.frame import DataFrame

from typing import List, Optional, Any


class LumbaRandomForestRegressor:
    model: RandomForestRegressor

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe = dataframe
    
    def train_model(self, train_column_names: List[str], target_column_name: str, train_size: float = 0.8) -> dict:
        x = self.dataframe[train_column_names]
        y = self.dataframe[target_column_name]

        # check if the columns selected are valid for Random Forest process
        for col in x.columns:
            if y.dtype not in ["int64", "float64"] or x[col].dtype not in ["int64", "float64"]:
                return {
                    'error': 'Kolom yang boleh dipilih hanyalah kolom dengan data numerik saja. Silakan pilih kolom yang benar atau gunakan encoding pada data categorical.'
                }
            
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)

        rfr = RandomForestRegressor()
        rfr.fit(x_train, y_train)

        y_pred = rfr.predict(x_test)

        r2 = r2_score(y_true=y_test, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        mse = mean_squared_error(y_true=y_test, y_pred=y_pred)

        self.model = rfr

        return {
            'model': rfr,
            'r2_score': f'{r2:.4f}',
            'mean_absolute_error': f'{mae:.4f}',
            'mean_squared_error': f'{mse:.4f}',
        }
    
    def get_model(self) -> Optional[RandomForestRegressor]:
        try:
            return self.model
        except AttributeError:
            return None

    def predict(self, data_target: Any) -> Any:
        rfr = self.get_model()
        y_pred = rfr.predict(data_target)

        return y_pred