"""
@author: Dylan Chen

For regression: f_regression, mutual_info_regression
For classification: chi2 (categorical), f_classif (numerical), mutual_info_classif (both)

"""

import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.stats import pearsonr
from sklearn.feature_selection import chi2, f_classif, f_regression
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.multiclass import type_of_target


def pairwise_corr(df, limit=0.5):
    corr_df = df.corr()
    indices = np.where(abs(corr_df) > limit)
    corr_list = [(corr_df.index[x], corr_df.columns[y], corr_df.ix[x, y]) for x, y in zip(*indices) if x != y and x < y]
    corr_df = pd.DataFrame.from_records(corr_list, columns=['var1', 'var2', 'pearson_corr']).sort_values('pearson_corr', ascending=False)
    return corr_df


class WOE:
    def __init__(self):
        self._WOE_MIN = -20
        self._WOE_MAX = 20

    def woe(self, X, y, event=1):
        '''
        Calculate woe of each feature category and information value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable which should be binary
        :param event: value of binary stands for the event to predict
        :return: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
                 numpy array of information value of each feature
        '''
        self.check_target_binary(y)
        X1 = self.feature_discretion(X)

        res_woe = []
        res_iv = []
        for i in range(0, X1.shape[-1]):
            x = X1[:, i]
            woe_dict, iv1 = self.woe_single_x(x, y, event)
            res_woe.append(woe_dict)
            res_iv.append(iv1)
        return np.array(res_woe), np.array(res_iv)

    def woe_single_x(self, x, y, event=1):
        '''
        calculate woe and information for a single feature
        :param x: 1-D numpy starnds for single feature
        :param y: 1-D numpy array target variable
        :param event: value of binary stands for the event to predict
        :return: dictionary contains woe values for categories of this feature
                 information value of this feature
        '''
        self.check_target_binary(y)

        event_total, non_event_total = self.count_binary(y, event=event)
        # x_labels = np.unique(x)
        x_labels = pd.Series(x).unique()
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y.iloc[np.where(x == x1)[0]]
            event_count, non_event_count = self.count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = self._WOE_MIN
            elif rate_non_event == 0:
                woe1 = self._WOE_MAX
            else:
                woe1 = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
        return woe_dict, iv

    def woe_replace(self, X, woe_arr):
        '''
        replace the explanatory feature categories with its woe value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param woe_arr: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
        :return: the new numpy array in which woe values filled
        '''
        if X.shape[-1] != woe_arr.shape[-1]:
            raise ValueError('WOE dict array length must be equal with features length')

        res = np.copy(X).astype(float)
        idx = 0
        for woe_dict in woe_arr:
            for k in woe_dict.keys():
                woe = woe_dict[k]
                res.iloc[:, idx][np.where(res.iloc[:, idx] == k)[0]] = woe * 1.0
            idx += 1

        return res

    def combined_iv(self, X, y, masks, event=1):
        '''
        calcute the information vlaue of combination features
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable
        :param masks: 1-D numpy array of masks stands for which features are included in combination,
                      e.g. np.array([0,0,1,1,1,0,0,0,0,0,1]), the length should be same as features length
        :param event: value of binary stands for the event to predict
        :return: woe dictionary and information value of combined features
        '''
        if masks.shape[-1] != X.shape[-1]:
            raise ValueError('Masks array length must be equal with features length')

        x = X.iloc[:, np.where(masks == 1)[0]]
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(self.combine(x.iloc[i, :]))

        dumy = np.array(tmp)
        # dumy_labels = np.unique(dumy)
        woe, iv = self.woe_single_x(dumy, y, event)
        return woe, iv

    def combine(self, list):
        res = ''
        for item in list:
            res += str(item)
        return res

    def count_binary(self, a, event=1):
        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count

    def check_target_binary(self, y):
        '''
        check if the target variable is binary, raise error if not.
        :param y:
        :return:
        '''
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise ValueError('Label type must be binary')

    def feature_discretion(self, X):
        '''
        Discrete the continuous features of input data X, and keep other features unchanged.
        :param X : numpy array
        :return: the numpy array in which all continuous features are discreted
        '''
        temp = []
        for i in range(0, X.shape[-1]):
            x = X.iloc[:, i]
            x_type = type_of_target(list(x))
            if x_type == 'continuous':
                x1 = self.discrete(x)
                temp.append(x1)
            else:
                temp.append(x)
        return np.array(temp).T

    def discrete(self, x):
        '''
        Discrete the input 1-D numpy array using 5 equal percentiles
        :param x: 1-D numpy array
        :return: discreted 1-D numpy array
        '''
        res = np.array([0] * x.shape[-1], dtype=int)
        for i in range(5):
            point1 = stats.scoreatpercentile(x, i * 20)
            point2 = stats.scoreatpercentile(x, (i + 1) * 20)
            x1 = x.iloc[np.where((x >= point1) & (x <= point2))]
            mask = np.in1d(x, x1)
            res[mask] = (i + 1)
        return res

    @property
    def WOE_MIN(self):
        return self._WOE_MIN
    @WOE_MIN.setter
    def WOE_MIN(self, woe_min):
        self._WOE_MIN = woe_min
    @property
    def WOE_MAX(self):
        return self._WOE_MAX
    @WOE_MAX.setter
    def WOE_MAX(self, woe_max):
        self._WOE_MAX = woe_max


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
        if len(r) > 2:
            df = [('feature_name', self.feature_name), (r[2], r[0]), ('p_value', r[1])]
            df = pd.DataFrame.from_items(df)
            df.sort_values('p_value', inplace=True)
        elif len(r) == 2:
            df = r[0]
            labels = ['feature_name', r[1], 'p_value']
            df = pd.DataFrame.from_records(df, columns=labels)
            df.sort_values('p_value', inplace=True)
        else:
            df = [('feature_name', self.feature_name), ('MutualInfo', r[0])]
            df = pd.DataFrame.from_items(df)
            df.sort_values('MutualInfo', ascending=False, inplace=True)
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
        score_name = 'chi2'
        return score, p_value, score_name


class FscoreSelect(FeatureSelect):
    def __init__(self, x, y, feature_name, model_type):
        self.type = model_type
        super().__init__(x, y, feature_name)

    def select_algo(self):
        if self.type == 'class':
            score, p_value = f_classif(self.x, self.y)
            score_name = 'Fscore'
            return score, p_value, score_name

        elif self.type == 'regression':
            """
            The cross correlation between each regressor and the target is computed,
            that is, ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) *
            std(y)).
            """
            score, p_value = f_regression(self.x, self.y)
            score_name = 'Fscore'
            return score, p_value, score_name


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


class PearsonCorr(FeatureSelect):
    def __init__(self, x, y, feature_name):
        super().__init__(x, y, feature_name)

    def select_algo(self):
        corr_list = []
        for name, col in zip(self.feature_name, self.x.T):
            corr = pearsonr(col, self.y)
            corr_list.append([name] + list(corr))
        score_name = 'pearson_corr'
        return corr_list, score_name