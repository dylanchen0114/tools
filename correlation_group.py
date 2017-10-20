# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import pandas as pd
from sklearn.cluster import DBSCAN


class CorrGroup:
    def __init__(self, df):
        self.corr_matrix = 1 - abs(df.corr().fillna(0))

    def fit_predict(self, esp, min_points):
        model = DBSCAN(eps=esp, min_samples=min_points, metric='precomputed')
        _labels = model.fit_predict(self.corr_matrix)
        return _labels

    def transform(self, esp, min_points):
        _labels = self.fit_predict(esp, min_points)
        var_names = list(self.corr_matrix.index)

        trans_df = pd.DataFrame({'feature_names': var_names, 'labels': _labels}).sort_values('labels', ascending=False)

        return trans_df

