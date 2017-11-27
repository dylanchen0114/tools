# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

from pkl_utils import save_pickle

import numpy as np
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

        """
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(x_train))))
        F_ij = np.dot((x_train / denom[:, None]).T, x_train)  # Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij)  # Inverse Information Matrix
        sigma_estimates = np.array([np.sqrt(Cramer_Rao[i, i]) for i in range(Cramer_Rao.shape[0])])  # sigma for each coefficient
        z_scores = self.model.coef_[0] / sigma_estimates  # z-score for each model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]  # two tailed test for p-values
        """

        def log_likelihood(features, target, weights):
            scores = np.dot(features, weights)
            ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
            return ll

        """
        self.z_scores = z_scores
        self.p_values = p_values
        self.sigma_estimates = sigma_estimates
        self.F_ij = F_ij
        """

        save_pickle(self.model, '%s/scikit-learn-model-%s.pkl' % (directory, name))

        if hasattr(self.model, 'coef_'):
            _coef = self.model.coef_[0]
            coef_df = pd.DataFrame({'Var_Name': feature_names, 'Var_Coef': _coef})
            coef_df['Var_Coef_ABS'] = abs(coef_df['Var_Coef'])
            coef_df = coef_df.sort_values('Var_Coef_ABS', ascending=0)
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
        std_err = summary.bse
        z_score = summary.tvalues

        tmp_df = pd.DataFrame(
            {'feature': feature_names + ['const'], "pvals": pvals, "coeff": coef, "conf_lower": conf_lower,
             "conf_higher": conf_higher, 'std_err': std_err, 'z_score': z_score})

        # Reordering...
        results_df = tmp_df[['feature', "coeff", 'std_err', 'z_score', "pvals", "conf_lower", "conf_higher"]]
        results_df.to_csv('%s/SMLogit-coef-%s.csv' % (directory, name), index=None)

    def fit(self, x_train, y_train, x_eval=None, y_eval=None, seed=42, name=None, directory=None, feature_names=None, **kwa):
        x_train = pd.DataFrame(x_train, columns=feature_names)
        self.model = sm.Logit(y_train, sm.add_constant(x_train, prepend=False, has_constant=self.has_constant)).fit()
        print(self.model.summary())
        self.summary2df(self.model, name=name, feature_names=feature_names, directory=directory)
        save_pickle(self.model, '%s/SMLogit-model-%s.pkl' % (directory, name))

    def predict(self, x):
        return self.model.predict(sm.add_constant(x, prepend=False, has_constant=self.has_constant))