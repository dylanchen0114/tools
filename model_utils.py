# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

from pkl_utils import save_pickle

import pandas as pd
import statsmodels.api as sm
import xgbfir
import xgboost as xgb


class BaseAlgo(object):

    def fit_predict(self, train, val=None, test=None, **kwa):
        self.fit(train[0], train[1], val[0] if val else None, val[1] if val else None, **kwa)

        if val is None:
            return self.predict(test[0])
        else:
            return self.predict(val[0]), self.predict(test[0])


class Xgb(BaseAlgo):

    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }

    def __init__(self, params, n_iter=400):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.n_iter = n_iter

    def fit(self, x_train, y_train, x_eval=None, y_eval=None, seed=42, feature_names=None, name=None, directory=None, xgbfir_tag=1):

        params = self.params.copy()
        params['seed'] = seed

        dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)

        if x_eval is None:
            watchlist = [(dtrain, 'train')]
        else:
            deval = xgb.DMatrix(x_eval, label=y_eval, feature_names=feature_names)
            watchlist = [(deval, 'eval'), (dtrain, 'train')]

        self.model = xgb.train(params, dtrain, self.n_iter, watchlist, verbose_eval=50)
        self.model.dump_model('%s/xgb-%s.dump' % (directory, name), with_stats=True)
        self.feature_names = feature_names

        if xgbfir_tag:
            xgbfir.saveXgbFI(self.model, feature_names=self.feature_names, OutputXlsxFile='%s/xgb-%s-feature-importance.xlsx' % (directory, name))

    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X, feature_names=self.feature_names))


class Sklearn(BaseAlgo):
    def __init__(self, model):
        self.model = model

    def fit(self, x_train, y_train, x_eval=None, y_eval=None, seed=42, feature_names=None, name=None, directory=None, eval_func=None, **kwa):
        self.model.fit(x_train, y_train)

        if x_eval is not None and hasattr(self.model, 'staged_predict'):
            for i, p_eval in enumerate(self.model.staged_predict(x_eval)):
                print("Iter %d score: %.5f" % (i, eval_func(y_eval, p_eval)))

        save_pickle(self.model, '%s/scikit-learn-model-%s.pkl' % (directory, name))

        if hasattr(self.model, 'coef_'):
            _coef = self.model.coef_[0]
            coef_df = pd.DataFrame({'Var_Name': feature_names, 'Var_Coef': _coef})
            coef_df.to_csv('%s/feature_coef_%s.csv' % (directory, name), index=None)

    def predict(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)


class SMLogit(BaseAlgo):
    def __init__(self, has_constant):
        self.has_constant = has_constant

    def summary2df(self, summary, name, feature_names, directory):
        pvals = summary.pvalues
        coef = summary.params
        conf_lower = summary.conf_int()[0]
        conf_higher = summary.conf_int()[1]

        tmp_df = pd.DataFrame(
            {'feature': feature_names + ['const'], "pvals": pvals, "coeff": coef, "conf_lower": conf_lower,
             "conf_higher": conf_higher})

        # Reordering...
        results_df = tmp_df[['feature', "coeff", "pvals", "conf_lower", "conf_higher"]]
        results_df.to_csv('%s/SMLogit-coef-%s.csv' % (directory, name), index=None)

    def fit(self, x_train, y_train, x_eval=None, y_eval=None, seed=42, name=None, directory=None, feature_names=None, **kwa):
        x_train = pd.DataFrame(x_train, columns=feature_names)
        self.model = sm.Logit(y_train, sm.add_constant(x_train, prepend=False, has_constant=self.has_constant)).fit()

        self.summary2df(self.model, name=name, feature_names=feature_names, directory=directory)
        save_pickle(self.model, '%s/SMLogit-model-%s.pkl' % (directory, name))

    def predict(self, x):
        return self.model.predict(sm.add_constant(x, prepend=False, has_constant=self.has_constant))