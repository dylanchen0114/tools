# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

For regression: f_regression, mutual_info_regression
For classification: chi2 (categorical), f_classif (numerical), mutual_info_classif (both)

"""

import numpy as np
import pandas as pd

from sklearn.feature_selection import chi2, f_classif, f_regression
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class FeatureSelect:
    def __init__(self, x, y, feature_name):
        """
        x and y can be array, series or dataFrame

        :param x: input features
        :param y: target y
        :param feature_name: list, names of features
        """

        self.feature_name = feature_name

        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            self.x = x
            self.y = y

        if isinstance(x, (pd.Series, pd.DataFrame)) and isinstance(y, (pd.Series, pd.DataFrame)):
            self.x = x.values
            self.y = y.values

    def score(self):
        r = self.select_algo()
        if len(r) > 1:
            df = pd.DataFrame({'feature_name': self.feature_name, 'score': r[0], 'p_value': r[1]})
            df.sort_values('p_value', inplace=True)
        else:
            df = pd.DataFrame({'feature_name': self.feature_name, 'score': r[0]})
            df.sort_values('score', ascending=False, inplace=True)
        return df


class Chi2Select(FeatureSelect):
    """
    This score can be used to select the n_features features with the
    highest values for the test chi-squared statistic from X, which must
    contain only non-negative features such as booleans or frequencies
    (e.g., term counts in document classification), relative to the classes.

    Recall that the chi-square test measures dependence between stochastic
    variables, so using this function "weeds out" the features that are the
    most likely to be independent of class and therefore irrelevant for
    classification.
    """

    def __init__(self, x, y, feature_name):
        super().__init__(x, y, feature_name)

    def select_algo(self):
        score, p_value = chi2(self.x, self.y)
        return score, p_value


class FscoreSelect(FeatureSelect):
    def __init__(self, x, y, feature_name, model_type):
        self.type = model_type
        super().__init__(x, y, feature_name)

    def select_algo(self):
        if self.type == 'class':
            score, p_value = f_classif(self.x, self.y)
            return score, p_value

        elif self.type == 'regression':
            """
            The cross correlation between each regressor and the target is computed,
            that is, ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) *
            std(y)).
            """
            score, p_value = f_regression(self.x, self.y)
            return score, p_value


class MutualInfo(FeatureSelect):
    def __init__(self, x, y, feature_name, model_type, discrete_features='auto', n_neighbors=3, random_state=114):
        self.type = model_type
        self.discrete_features = discrete_features
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        super().__init__(x, y, feature_name)

    def select_algo(self):
        if self.type == 'class':
            score = mutual_info_classif(self.x, self.y, discrete_features=self.discrete_features,
                                        n_neighbors=self.n_neighbors, random_state=self.random_state)
            return [score]

        elif self.type == 'regression':
            score = mutual_info_regression(self.x, self.y, discrete_features=self.discrete_features,
                                           n_neighbors=self.n_neighbors, random_state=self.random_state)
            return [score]


if __name__ == '__main__':
    test = pd.DataFrame({'x1': [2, 2, 2, 2], 'x2': [1, 2, 3, 1], 'y': [4, 4, 4, 4]})

    print(MutualInfo(test[['x1', 'x2']], test['y'], ['x1', 'x2'], 'regression', random_state=1).score())
    print(mutual_info_regression(test[['x1', 'x2']], test['y'], random_state=1))
