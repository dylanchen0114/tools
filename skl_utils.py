# -*- coding: utf-8 -*-

from itertools import combinations, product

import numpy as np
import sklearn.ensemble
import sklearn.neighbors
import sklearn.svm
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from mst_clustering import MSTClustering
from sklearn import metrics

from my_tools import dist_utils


def silhouette_metric(X, method='Kmeans', k_range=10):
    label_list = []
    silhouette_score = []
    for k in range(2, k_range):
        rd_score = []
        for rd in range(10):
            labels = KMeans(n_clusters=k, random_state=2017 + k*rd).fit_predict(X)
            label_list.append(labels)
            score = metrics.silhouette_score(X=X, labels=labels)
            rd_score.append(score)
        best_score = max(rd_score)
        silhouette_score.append(best_score)
    return silhouette_score, label_list


class SVR:
    def __init__(self, kernel='rbf', degree=3, gamma='auto', C=1.0, 
                epsilon=0.1, normalize=True, cache_size=2048):
        svr = sklearn.svm.SVR(kernel=kernel, degree=degree, 
                            gamma=gamma, C=C, epsilon=epsilon)
        if normalize:
            self.model = Pipeline([('ss', StandardScaler()), ('svr', svr)])
        else:
            self.model = svr
            
    def __str__(self):
        return "SVR"

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred


class LinearSVR:
    def __init__(self, epsilon=0.0, C=1.0, loss='epsilon_insensitive', 
                random_state=None, normalize=True):
        lsvr = sklearn.svm.LinearSVR(epsilon=epsilon, C=C, 
                    loss=loss, random_state=random_state)
        if normalize:
            self.model = Pipeline([('ss', StandardScaler()), ('lsvr', lsvr)])
        else:
            self.model = lsvr

    def __str__(self):
        return "LinearSVR"

    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred


class KNNRegressor:
    def __init__(self, n_neighbors=5, weights='uniform', leaf_size=30, 
                metric='minkowski', normalize=True):
        if metric == 'cosine':
            metric = lambda x,y: dist_utils._cosine_sim(x, y)
        knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, 
            leaf_size=leaf_size, metric=metric)
        if normalize:
            self.model = Pipeline([('ss', StandardScaler()), ('knn', knn)])
        else:
            self.model = knn

    def __str__(self):
        return "KNNRegressor"

    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred


class AdaBoostRegressor:
    def __init__(self, base_estimator=None, n_estimators=50, max_features=1.0,
                max_depth=6, learning_rate=1.0, loss='linear', random_state=None):
        if base_estimator and base_estimator == 'etr':
            base_estimator = ExtraTreeRegressor(max_depth=max_depth,
                                        max_features=max_features)
        else:
            base_estimator = DecisionTreeRegressor(max_depth=max_depth,
                                        max_features=max_features)

        self.model = sklearn.ensemble.AdaBoostRegressor(
                                    base_estimator=base_estimator,
                                    n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    random_state=random_state,
                                    loss=loss)

    def __str__(self):
        return "AdaBoostRegressor"

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred


class RandomRidge:
    def __init__(self, alpha=1.0, normalize=True, poly=False,
                    n_estimators=10, max_features=1.0,
                    bootstrap=True, subsample=1.0,
                    random_state=2016):
        self.alpha = alpha
        self.normalize = normalize
        self.poly = poly
        self.n_estimators = n_estimators
        if isinstance(max_features, float):
            assert max_features > 0 and max_features <= 1
        self.max_features = max_features
        self.bootstrap = bootstrap
        assert subsample > 0 and subsample <= 1
        self.subsample = subsample
        self.random_state = random_state
        self.ridge_list = [0]*self.n_estimators
        self.feature_idx_list = [0]*self.n_estimators

    def __str__(self):
        return "RandomRidge"

    def _random_feature_idx(self, fdim, random_state):
        rng = np.random.RandomState(random_state)
        if isinstance(self.max_features, int):
            size = min(fdim, self.max_features)
        else:
            size = int(fdim * self.max_features)
        idx = rng.permutation(fdim)[:size]
        return idx

    def _random_sample_idx(self, sdim, random_state):
        rng = np.random.RandomState(random_state)
        size = int(sdim * self.subsample)
        if self.bootstrap:
            idx = rng.randint(sdim, size=size)
        else:
            idx = rng.permutation(sdim)[:size]
        return idx

    def fit(self, X, y):
        sdim, fdim = X.shape
        for i in range(self.n_estimators):
            ridge = Ridge(alpha=self.alpha, normalize=self.normalize, random_state=self.random_state)
            fidx = self._random_feature_idx(fdim, self.random_state+i*100)
            sidx = self._random_sample_idx(sdim, self.random_state+i*10)
            X_tmp = X[sidx][:,fidx]
            if self.poly:
                X_tmp = PolynomialFeatures(degree=2).fit_transform(X_tmp)[:,1:]
            ridge.fit(X_tmp, y[sidx])
            self.ridge_list[i] = ridge
            self.feature_idx_list[i] = fidx
        return self

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], self.n_estimators))
        for i in range(self.n_estimators):
            fidx = self.feature_idx_list[i]
            ridge = self.ridge_list[i]
            X_tmp = X[:,fidx]
            if self.poly:
                X_tmp = PolynomialFeatures(degree=2).fit_transform(X_tmp)[:,1:]
            y_pred[:,i] = ridge.predict(X_tmp)
        y_pred = np.mean(y_pred, axis=1)
        return y_pred


class SklDBSCAN:
    def __init__(self, eps, min_sample, distance_frame, metric):
        self.metric = metric
        self.eps = eps
        self.min_sample = min_sample
        self.distance_matrix = distance_frame.values
        self.db = self.clustering()
        self.score = np.sum(self.ISSE()) / np.sum(self.OSSE())

    def clustering(self):
        db = DBSCAN(eps=self.eps, min_samples=self.min_sample, metric=self.metric, n_jobs=-1)
        db.fit(self.distance_matrix)

        return db

    def ISSE(self):
        labels = self.db.labels_
        cluster_list = []
        for label in set(labels):
            cluster_set = []
            lable_index = list(
                combinations([lable_index for lable_index, each in enumerate(labels) if each == label], 2))
            for row_col in lable_index:
                cluster_set.append(self.distance_matrix[row_col])
            cluster_mean = np.mean(cluster_set)
            cluster_list.append(cluster_mean)

            # print('Cluster %s: With-in Mean Similarity %s' % (label, cluster_mean))

        return cluster_list

    def OSSE(self):
        labels = self.db.labels_
        cluster_list = []
        for label_combine in combinations(set(labels), 2):
            cross_cluster = []
            for label in label_combine:
                lable_index = [lable_index for lable_index, each in enumerate(labels) if each == label]
                cross_cluster.append(lable_index)
            product_cross = list(product(*cross_cluster))
            min_dis = min([self.distance_matrix[row_col] for row_col in product_cross])
            cluster_list.append(min_dis)

            # print('Cross Clusters %s: Cross Mean Similarity %s' % (label_combine, min_dis))

        return cluster_list


class MST:
    def __init__(self, cutoff, min_cluster_size, distance_frame):
        self.cutoff = cutoff
        # self.cutoff_scale = cutoff_scale
        self.min_cluster_size = min_cluster_size
        self.distance_matrix = distance_frame.values
        self.db = self.clustering()
        self.score = np.sum(self.ISSE()) / np.sum(self.OSSE())

    def clustering(self):
        db = MSTClustering(cutoff=self.cutoff, min_cluster_size=self.min_cluster_size, metric='precomputed')
        db = db.fit(self.distance_matrix)
        return db

    def ISSE(self):
        labels = self.db.labels_
        cluster_list = []
        for label in set(labels):
            cluster_set = []
            label_index = list(
                combinations([label_index for label_index, each in enumerate(labels) if each == label], 2))
            for row_col in label_index:
                cluster_set.append(self.distance_matrix[row_col])
            cluster_mean = np.mean(cluster_set)
            cluster_list.append(cluster_mean)
            # print('Cluster %s: With-in Mean Similarity %s' % (label, cluster_mean))
        return cluster_list

    def OSSE(self):
        labels = self.db.labels_
        cluster_list = []
        for label_combine in combinations(set(labels), 2):
            cross_cluster = []
            for label in label_combine:
                label_index = [label_index for label_index, each in enumerate(labels) if each == label]
                cross_cluster.append(label_index)
            product_cross = list(product(*cross_cluster))
            min_dis = min([self.distance_matrix[row_col] for row_col in product_cross])
            cluster_list.append(min_dis)
            # print('Cross Clusters %s: Cross Mean Similarity %s' % (label_combine, min_dis))
        return cluster_list
