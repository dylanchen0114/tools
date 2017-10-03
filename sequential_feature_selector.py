# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


class SklearnFeatureSelector:

    def __init__(self, model):
        self.model = model

    def run(self, x_train, y_train, k_features, floating, scoring, cv, out_path, forward=True, plot_figure=False):
        self.sequential_selector(x_train, y_train, k_features, floating, scoring, cv, forward=forward)
        self.evaluate_performance(out_path=out_path, plot_figure=plot_figure)

    def sequential_selector(self, x_train, y_train, k_features, floating, scoring='roc_auc', cv=0, forward=True):
        selector = SFS(self.model, k_features=k_features, forward=forward, floating=floating,
                       scoring=scoring, verbose=2, cv=cv, n_jobs=-1)

        # ensure train x and y is an array
        self.selector = selector.fit(x_train, y_train)

    def evaluate_performance(self, out_path, plot_figure=False):

        print('\nSaving Metric Dict ...')
        metric_dict = self.selector.get_metric_dict()
        tmp_df = pd.DataFrame.from_dict(metric_dict).T
        tmp_df.to_csv('%s/sequential_selector_metric_df.csv' % out_path)

        if plot_figure:
            print('\nSaving Plotting Figure ...')
            fig = plot_sfs(self.selector.get_metric_dict(), kind='std_dev')

            plt.ylim([0.5, 1])
            plt.grid()
            plt.show()









