# -*- coding: utf-8 -*-
#!/usr/bin/env python
__author__ = 'chendeqing'

import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# from sklearn.ensemble import VotingClassifier
# from sklearn.grid_search import GridSearchCV

# Gradient Boost
# import xgboost as xgb

from scipy.stats import pearsonr, ks_2samp
from datetime import datetime

import traceback
import dateutil
import re
import time
import os
from os.path import split, splitext, join
import sys
# sys.path.append('/home/risk_chendeqing/modules')

import pickle
import dumb_containers as dc


''' ID number '''
def get_id_age(id_s, ref_date=None):
    if ref_date is None:
        ref_date = datetime.now()
    
    age_birthdate = pd.to_datetime(id_s.apply(lambda i: str(i)[6:14]), errors='coerce')
    age = pd.Series(len(age_birthdate)*[np.nan], index=age_birthdate.index)
    age.loc[~pd.isnull(age_birthdate)] = age_birthdate.loc[~pd.isnull(age_birthdate)].apply(lambda bd: dateutil.relativedelta.relativedelta(ref_date, bd).years)
    return age
    
def gender_fr_id(id_no):
    try:
        gender = int(id_no[-2]) % 2
        if gender == 0:
            gender = 2
    except ValueError:
        gender = 0
    
    return gender

def get_id_gender(id_s):
    return id_s.apply(gender_fr_id)

''' '''
def isclose(a, b, rel_tol=1e-04, abs_tol=1e-6):
    try:
        return abs(a-b) <= np.max([rel_tol * np.max([abs(a), abs(b)], axis=0),
                                   abs_tol * np.ones(len(a))], axis=0)
    except TypeError:
        return abs(a-b) <= np.max([rel_tol * np.max([abs(a), abs(b)], axis=0),
                                   abs_tol], axis=0)

def is_contain_str(item, pat):
    '''
    Check if one string contains another, regex supported. False if either one is not a string.
    '''
    if (isinstance(item, basestring)
        and isinstance(pat, basestring)
        and re.search(pat, item) is not None):
        return True
    else:
        return False
    
def is_num(i):
    '''
    Check if a string is a number.
    '''
    try:
        float(i)
    except ValueError:
        return False
    else:
        return True

def cap_percent(df, cols=None, low_percent=None, up_percent=None, inplace=False):
    '''
    Cap data with percenttile.
    '''
    if cols is None:
        print("NOTE all columns, including string columns, will be processed")
        cols = df.columns
    
    if not inplace:
        data = df.copy()
    else:
        data = df
    
    for c in cols:
        try:
            if low_percent is not None:
                floor = data[c].quantile(low_percent, interpolation='lower')
                data.ix[data[c] < floor, c] = floor
            
            if up_percent is not None:
                ceil = data[c].quantile(up_percent, interpolation='higher')
                data.ix[data[c] > ceil, c] = ceil
        except KeyError:
            print("---> No column '{}' found".format(c))
            
    if not inplace:
        return data

def cap_value(df, cols=None, low_value=None, up_value=None, inplace=False):
    '''
    Cap data with value.
    '''
    if cols is None:
        print("NOTE all columns, including string columns, will be processed")
        cols = df.columns
    
    if not inplace:
        data = df.copy()
    else:
        data = df
        
    for c in cols:
        try:
            if low_value is not None:
                data.ix[data[c] < low_value, c] = low_value
            
            if up_value is not None:
                data.ix[data[c] > up_value, c] = up_value
        except KeyError:
            print("---> No column '{}' found".format(c))
    
    if not inplace:
        return data

def get_bin_range(ref_table, var=None):
    '''
    Get bin range from reference table.
    
    Var_Value  Ref_Value     Var_Name
    -inf_0.5   0.179195  count_phone
     1.5_inf  -0.096664  count_phone
     0.5_1.5   0.049708  count_phone
        base  -3.933759  count_phone
          IV   0.012190  count_phone

    >>> get_bin_range(df_numeric_ref_table, 'count_phone')
    [-inf, 0.5, 1.5, inf]
    '''
    if var is None:
        # ref_table has only one variable, and no Var_Name
        vv = ref_table['Var_Value']
    else:
        vv = ref_table.ix[ref_table['Var_Name'] == var, 'Var_Value']
    
    b_lst = []
    for v in vv:
        if 'base' == v or 'iv' == v.lower():
            continue
        t_lst = v.split('_')
        b_lst.extend([float(b) for b in t_lst])
    
    b_lst = sorted(list(set(b_lst)))
    b_lst[-1] += .1
    return b_lst


def conv_cols_to_float(data, verbose=False):
    '''
    Convert columns to float if possible
    '''
    if not verbose:
        print("---> Error message suppressed.")
        
    for c in data.columns:
        try:
            data[c] = data[c].astype(float)
        except ValueError:
            if verbose:
                print("---> convert {} to float failed".format(c))
        except TypeError:
            print("{} is not a string or a number".format(c))
    
    return data

def corr_pcorr(data, target, bn):
    t0 = time.clock()
    # Correlation matrix
    data.corr().to_csv("{0}_corr_{1}.csv".format(bn, str(datetime.now().date())))

    # Correlation with p-value
    if type(target) == str:
        y = data[target]
    else:
        # A series
        y = target
        
    pcorr = []
    for v, t in zip(data.columns, data.dtypes):
        if object != t and '<M8[ns]' != t:
    #         print v, t
            c, p = pearsonr(data[v].fillna(0), y)
            pcorr.append({"var": v, "corr": c, "p-value": p})

    pcorr_df = pd.DataFrame(pcorr)
    pcorr_df = pcorr_df.sort_values(by="p-value").reindex_axis(['var', "corr", "p-value"], axis=1)
    print(pcorr_df.head())
    if bn:
        pcorr_df.to_csv("{0}_pcorr_{1}.csv".format(bn, str(datetime.now().date())), index=False)
    print("{0}s elapsed".format(time.clock()-t0))
    return pcorr_df

# def xgb_model(x_train, target, xgb_model=None, bn=None, toplot=True):
    # ### XGBoost
    # if type(target) == str:
        # y_train = x_train[target]
    # else:
        # y_train = target
        
    # dtrain = xgb.DMatrix(x_train, y_train, missing=np.nan)
    
    # is_train = False
    # if not xgb_model or isinstance(xgb_model, dict):
        # print("---> Fit XGB model on input data")
        # if isinstance(xgb_model, dict):
            # params = xgb_model
        # else:
            # params = {'max_depth':2,
                      # 'eta':0.8,
                      # 'silent':1,
                      # 'objective':'binary:logistic',
                      # 'eval_metric':'auc'}
        
        # is_train = True
        # xgb_model = xgb.train(params=params, dtrain=dtrain)
    # else:
        # print("---> Predict with XGB model")
    
    # predictions = xgb_model.predict(dtrain)
    # actuals = y_train
    # fpr, tpr, threshold = roc_curve(actuals, predictions)
    
    # # if toplot:
        # # ROC
        # # fpr, tpr, threshold = roc_curve(actuals, predictions)
        # # auc_ = auc(fpr, tpr)
        # # ks_ = max(tpr-fpr)

        # # plt.plot(fpr, tpr)
        # # plt.title("ROC: AUC={0:.4f}, KS={1:.3f}".format(auc_, ks_))
        
    # dc.evaluate_performance(actuals.values, predictions, toplot=toplot)
    
    # # Output when training
    # xgb_i = None
    # if is_train:
        # xgb_i = pd.DataFrame.from_dict(xgb_model.get_fscore(), orient='index').reset_index().rename(columns={'index':'var', 0:'importance'}).sort_values(by='importance', ascending=False)
        # xgb_i.head()
        
        # if bn:
            # xgb_i.to_csv("{0}_xgb_vars_{1}.csv".format(bn, str(datetime.now().date())),
                         # index=False, encoding='utf-8')
    
    # return xgb_model, xgb_i

def lr_model(x_train, y_train, lm=None, has_constant='add', toplot=True, rtn_pred=False):
    if not lm:
        print("---> Fit LR model on input data")
        lm = sm.Logit(y_train,
                      sm.add_constant(x_train, prepend=False, has_constant=has_constant)).fit()
    else:
        print("---> Predict with LR model")
    
    predictions = lm.predict(sm.add_constant(x_train, prepend=False, has_constant=has_constant))
    actuals = y_train

    if toplot:
        # ROC
        # fpr, tpr, threshold = roc_curve(actuals, predictions)
        # auc_ = auc(fpr, tpr)
        # ks_ = max(tpr-fpr)

        # plt.plot(fpr, tpr)
        # plt.title("ROC: AUC={0:.4f}, KS={1:.3f}".format(auc_, ks_))
        
        try:
            dc.evaluate_performance(actuals.values, predictions)
        except AttributeError:
            print("---> Failed to do evaluation")
        
    if not rtn_pred:
        return lm
    else:
        return lm, pd.Series(data=predictions, index=x_train.index)

def l1_model(x_train, y_train, C, bn=None):
    x_train_s = sm.add_constant(x_train.fillna(0), prepend=False, has_constant='add')

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train_s)

    clf_l1_LR = LogisticRegression(C=C, penalty='l1')
    clf_l1_LR.fit(x_train_s, y_train)
    coef_l1_LR = clf_l1_LR.coef_.ravel()
    print(coef_l1_LR[:10])

    # coef matrix
    coef = coef_l1_LR
    if x_train.shape[1] != len(coef_l1_LR):
        coef = coef_l1_LR[:-1]
    
    cc = pd.DataFrame({'var':x_train.columns, 'coef':coef, 'coef_abs':abs(coef)})
    cc = cc.sort_values(by="coef_abs", ascending=False).reindex_axis(['var', 'coef', 'coef_abs'], axis=1)
    
    if bn:
        cc.to_csv("{0}_coef_vars_{1:.3f}_{2}.csv".format(bn, C, str(datetime.now().date())), index=False)
        
    return clf_l1_LR, cc, scaler

# 计算psi统计量
# As a rule of thumb
# a PSI<0.1 indicates minimal change in the population.
# 0.1 to 0.2 indicates changes that might warrant further investigation,
# and a PSI >0.2 indicates a significant change in the population.
def calc_psi(df_ref, df2, var, max_bins=100):
    df = df_ref
    if len(df[var].unique()) == 1:
        iv = np.nan
    else:
        # create bucket
        if len(df[var].unique()) < max_bins:
            uvalue = np.sort(df[var].astype(float).unique())
            uvdiff = np.append(np.diff(uvalue).astype(float)/2, 0)
            uvbucket = np.append(uvalue.min(), uvalue + uvdiff)
        else:
            uvalue = np.empty(0)
            for i in np.arange(max_bins+1):
                try:
                    uvalue = np.unique((np.append(uvalue, df[var].quantile(float(i)/float(max_bins)))))
                except:
                    pass
            uvdiff = np.append(np.diff(uvalue).astype(float)/2, 0)
            uvbucket = np.append(uvalue.min(), uvalue + uvdiff)
        uvbucket[0] = -np.inf
        uvbucket[-1] = np.inf

        df_ref[var+'_bin'] = [tuple([float(j) for j in i.strip('([]').split(',')]) for i in np.array(pd.cut(df_ref[var].astype(float), uvbucket, retbins=True, include_lowest=True)[0])]
        df2[var+'_bin'] = [tuple([float(j) for j in i.strip('([]').split(',')]) for i in np.array(pd.cut(df2[var].astype(float), uvbucket, retbins=True, include_lowest=True)[0])]
        ds_ref = pd.DataFrame(df_ref[var+'_bin'].value_counts().reset_index())
        ds2 = pd.DataFrame(df2[var+'_bin'].value_counts().reset_index())
        ds_ref.rename(columns={'index': 'bins',
                            var+'_bin': 'count'}, inplace=True)
        ds2.rename(columns={'index': 'bins',
                            var+'_bin': 'count'}, inplace=True)

        ds = pd.merge(ds_ref, ds2, how='left', left_on=['bins'], right_on=['bins'])

        bad_dist = ds['count_x']/ds['count_x'].sum()
        good_dist = ds['count_y']/ds['count_y'].sum()

        try:
            # bad_dist = bad_dist.apply(lambda x: 0.0001 if x == 0 else x)
            iv_bin = (bad_dist - good_dist) * \
                                 (bad_dist / good_dist).apply(lambda x: math.log(x))
            iv = iv_bin.sum()
        except ZeroDivisionError:
            iv = 1
        
    return iv

# 计算样本变量KS差异
def calc_sample_ks(df_ref, df2, var=None, alpha=None):
    if var is None:
        ks, ks_pval = ks_2samp(df_ref, df2)
    else:
        try:
            ks, ks_pval = ks_2samp(df_ref[var], df2[var])
        except KeyError:
            err = traceback.format_exc()
            print(err)
            return None, None, None, None
            
    # alpha: 0.10 	0.05 	0.025 	0.01 	0.005 	0.001
    # c(alpha): 1.22 	1.36 	1.48 	1.63 	1.73 	1.95
    c_alpha = {0.1: 1.22,
               0.05: 1.36,
               0.025: 1.48,
               0.01: 1.63,
               0.005: 1.73,
               0.001: 1.95,}
    if alpha is None:
        alpha = 0.05
        
    thresh = c_alpha[alpha] * math.sqrt(1.0 * (df_ref.shape[0] + df2.shape[0]) / df_ref.shape[0] / df2.shape[0])
    if ks > thresh:
        # Reject null hypothesis (different distribution)
        is_same_dist = False
    else:
        # Accept null hypothesis (same distribution)
        is_same_dist = True
        
    return ks, ks_pval, thresh, is_same_dist
    