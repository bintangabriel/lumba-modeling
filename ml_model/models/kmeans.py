# https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a

from sklearn.cluster import KMeans

from pandas.core.frame import DataFrame

from typing import Any, Optional, List

class LumbaKMeans:
    model: KMeans

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe = dataframe

    def train_model(self, train_column_names: List[str], k: int = 2) -> dict:
        x = self.dataframe[train_column_names]

        # check if the columns selected are valid for K-Means process
        for col in x.columns:
            if x[col].dtype not in ["int64", "float64"]:
                return {
                    'error': 'Kolom yang boleh dipilih hanyalah kolom dengan data numerik saja. Silakan pilih kolom yang benar atau gunakan encoding pada data categorical.'
                }
        
        km_model = KMeans(n_clusters=k)
        y_kmeans= km_model.fit_predict(x)

        self.model = km_model

        return {
            'model': km_model,
            'labels_predicted': y_kmeans,
        }

    def get_model(self) -> Optional[KMeans]:
        try:
            return self.model
        except AttributeError:
            return None

    