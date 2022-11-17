from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from pandas.core.frame import DataFrame

from typing import Any

class LumbaDecisionTreeClassifier:
    model: DecisionTreeClassifier

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe = dataframe

    def train_model(self):
        pass

    def get_model(self) -> DecisionTreeClassifier:
        return self.model

    def predict(self, data_target: Any) -> Any:
        pass