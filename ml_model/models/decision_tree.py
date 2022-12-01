from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pandas.core.frame import DataFrame

from typing import Any, List, Optional

class LumbaDecisionTreeClassifier:
    model: DecisionTreeClassifier

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe = dataframe

    def train_model(self, train_column_names: List[str], target_column_name: str, train_size: float = 0.8) -> dict:
        x = self.dataframe[train_column_names]
        y = self.dataframe[target_column_name]
        
        # check if the columns selected are valid for Decision Tree process
        for col in x.columns:
            if y.dtype not in ["int64", "float64"] or x[col].dtype not in ["int64", "float64"]:
                return {
                    'error': 'Kolom yang boleh dipilih hanyalah kolom dengan data numerik saja. Silakan pilih kolom yang benar atau gunakan encoding pada data categorical.'
                }
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)

        dt = DecisionTreeClassifier()
        dt.fit(x_train, y_train)

        y_pred = dt.predict(x_test)
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)

        self.model = dt

        return {
            'model': dt,
            'accuracy_score': f'{acc*100:.4f}'
        }

    def get_model(self) -> Optional[DecisionTreeClassifier]:
        try:
            return self.model
        except AttributeError:
            return None

    def predict(self, data_target: Any) -> Any:
        dt = self.get_model()
        y_pred = dt.predict(data_target)

        return y_pred