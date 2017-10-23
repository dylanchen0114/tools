import numpy as np
import pandas as pd
import os
import Model_old_customer.Tools.dumb_containers as dc
import Model_old_customer.Tools.Tools as tls
import matplotlib.pyplot as plt
import matplotlib

from datetime import datetime
from os.path import join


class WOE(object):
    """
    :parameter:
    'dpath':  woe生成文件储存根目录
    'cm':     woe生成文件储存根目录下文件夹
    'fname':  woe生成文件命名
    'target': woe所需目标列
    'save_pci': 是否保存woe转换图

    :returns:
    'x_train': woe转换后的训练集df（不保留原先的列）
    'x_test':  woe转换后的测试集df（不保留原先的列）
    'feature_names': woe生成的列名


        for example 1:

            # 在训练模型时fit训练集，transform验证集，直接得到两部分woe后的结果
            num_features = ['ovd_day_mean_p6m', 'order_rej_rto_p3m']
            cat_features = ['idnum_province']
            ds1 = {'features': Train_X, 'overdue': train_y}
            ds2 = {'features': Test_X, 'overdue': test_y}

            train_woe, test_woe, woe_features = WOE(cat_features=cat_features, num_features=num_features, dpath, cm,
                                                    fname, target, save_pic=1).woe_transform(ds1, ds2)

        for example 2:

            # 用已得到训练模型时产生的ref_table，transform需要的预测数据

            nominal_ref_table = pd.read_csv('/data/yezhenwei/OCODP/Overdue_15/Output/V1.2/woe_ref/overdue_nominal-woe-ref_2017-10-18.csv', encoding='utf8')
            numeric_ref_table = pd.read_csv('/data/yezhenwei/OCODP/Overdue_15/Output/V1.2/woe_ref/overdue_numerical-woe-ref_2017-10-18.csv')

            woe_x = woe_transform(pred[num_features + cat_features], nominal_ref_table, numeric_ref_table)

            # 这里不需要指定num_features和cat_features, 会根据nominal_ref_table和numeric_ref_table里有的列名生成相应的woe

    """
    def __init__(self, num_features=None, cat_features=None, dpath='/data/yezhenwei/OCODP/Overdue_15',
                 cm='/Output/woe/', fname='overdue', target='is_overdue', save_pic=0):
        self.dpath = dpath
        self.cm = cm
        self.fname = fname
        self.target = target

        if cat_features is None:
            self.cat_features = []
        else:
            self.cat_features = cat_features

        if num_features is None:
            self.num_features = []
        else:
            self.num_features = num_features

        self.save_pic = save_pic

    def woe_transform(self, ds1, ds2):

        x_train = ds1['features']
        train_target = pd.Series(ds1['overdue'])
        x_train[self.target] = train_target
        x_test = ds2['features']

        if self.cat_features:
            x_train, x_test = self.nominal_woe(x_train, x_test)
        if self.num_features:
            x_train, x_test = self.numeric_woe(x_train, x_test, save_pic=self.save_pic)

        x_train = x_train[[x for x in x_train.columns if x.endswith('woe')]]
        x_test = x_test[[x for x in x_test.columns if x.endswith('woe')]]
        feature_names = list(x_train.columns)

        return x_train, x_test, feature_names

    def nominal_woe(self, x_train, x_test):
        var_to_cwoe = self.cat_features
        ref_table = pd.DataFrame()
        iv_d = dict()

        b_stat_cvar_f = join(self.dpath + self.cm,
                             "{0}_cvar_bin_stats_{1}.csv".format(self.fname, str(datetime.now().date())))

        if os.path.exists(b_stat_cvar_f):
            os.remove(b_stat_cvar_f)

        var_no_cwoe = []
        print(len(var_to_cwoe))
        for v in var_to_cwoe:
            print('Now processing:', v)
            try:
                if len(x_train[v].unique()) > 700:
                    print("---> WARNING: too many unique values for {} to do chi-square merge".format(v))
                    var_no_cwoe.extend([v])
                    continue
            except KeyError:
                #         print("No column \"{}\" found".format(v))
                continue

            if v == self.target:
                var_no_cwoe.extend([v])
                continue

            rt, iv, b_stat = dc.calc_nominal_woe(x_train, v, self.target)
            if rt is None:
                print("---> WARNING: no reference table returned for {}".format(v))
                continue

            iv_d[v] = iv.sum()
            ref_table = pd.concat([ref_table, rt], axis=0, ignore_index=True)
            with open(b_stat_cvar_f, 'a') as bf:
                b_stat.to_csv(bf, sep='~', index=False, encoding='utf8')
                pd.DataFrame(b_stat[v]).T.to_csv(bf, sep=',', index=False, encoding='utf8', header=False)

        ref_table_cvar_f = join(self.dpath + self.cm,
                                "{0}_nominal-woe-ref_{1}.csv".format(self.fname, str(datetime.now().date())))
        ref_table.to_csv(ref_table_cvar_f, index=False)

        iv_cvar_f = join(self.dpath + self.cm,
                         "{0}_nominal-woe-ivs_{1}.csv".format(self.fname, str(datetime.now().date())))
        pd.DataFrame.from_dict(iv_d, orient='index').reset_index().rename(columns={'index': 'var', 0: 'iv'}). \
            sort_values('iv', ascending=False).to_csv(iv_cvar_f, index=False)

        # ## Set nominal woe
        ref_table = pd.read_csv(ref_table_cvar_f, encoding='utf8')
        dc.set_nominal_woe(x_train, ref_table)
        dc.set_nominal_woe(x_test, ref_table)

        return x_train, x_test

    def numeric_woe(self, x_train, x_test, save_pic=0):
        woe_numeric_vars = self.num_features
        b_stat_nvar_f = join(self.dpath + self.cm,
                             "{0}_nvar_bin_stats_{1}.csv".format(self.fname, str(datetime.now().date())))
        if os.path.exists(b_stat_nvar_f):
            os.remove(b_stat_nvar_f)

        iv = []
        df_numeric_ref_table = pd.DataFrame()
        print(len(woe_numeric_vars))
        for var in woe_numeric_vars:
            print('Now processing:', var)
            # ds,ref_table = main_get_ref_table(infile,var)
            ref_table, b_iv, b_stat = dc.main_get_numeric_ref_table(x_train, var, self.target, 20)
            iv.extend([ref_table['IV']])
            # df_ref_table_tmp = pd.DataFrame(ref_table.items(), columns=['Var_Value', 'Ref_Value'])
            df_ref_table_tmp = pd.DataFrame([[r, v] for r, v in ref_table.items()], columns=['Var_Value', 'Ref_Value'])
            df_ref_table_tmp['Var_Name'] = var
            df_numeric_ref_table = pd.concat((df_numeric_ref_table, df_ref_table_tmp), axis=0)

            with open(b_stat_nvar_f, 'a') as bf:
                b_stat.to_csv(bf, sep='~', index=False, encoding='utf8')
                bf.write(str(tls.get_bin_range(df_numeric_ref_table, var)))
                bf.write('\n')

        ref_table_nvar_f = join(self.dpath + self.cm,
                                "{0}_numerical-woe-ref_{1}.csv".format(self.fname, str(datetime.now().date())))
        df_numeric_ref_table.to_csv(ref_table_nvar_f, index=False)

        iv_nvar_f = join(self.dpath + self.cm,
                         "{0}_numerical-woe-ivs_{1}.csv".format(self.fname, str(datetime.now().date())))
        df_iv = pd.DataFrame({'var': woe_numeric_vars, 'iv': iv}).sort_values('iv', ascending=False)
        df_iv.to_csv(iv_nvar_f, index=False)

        # # Set WOE for numerical variables
        df_numeric_ref_table = pd.read_csv(ref_table_nvar_f)

        woe_numeric_vars = df_numeric_ref_table.Var_Name.unique()
        iv = []
        for var in woe_numeric_vars:
            print("Setting woe for {} ...".format(var))
            df_ref_table_var = df_numeric_ref_table[df_numeric_ref_table['Var_Name'] == var]
            ref_table = dict(zip(df_ref_table_var['Var_Value'], df_ref_table_var['Ref_Value']))
            iv.extend([ref_table['IV']])
            _ = dc.main_apply_numeric_ref_table(x_train, ref_table, var)
            _ = dc.main_apply_numeric_ref_table(x_test, ref_table, var)

        if save_pic:
            # Save woe images
            woe_path = os.path.join(self.dpath + self.cm, "woe_{0}".format(str(datetime.now().date())))

            if not os.path.isdir(woe_path):
                os.mkdir(woe_path)

            # Font used for Chinese in plot
            myfont = matplotlib.font_manager.FontProperties(fname='/home/yezhenwei/My_git/Model_old_customer/Tools/msyh.ttf')
            for var in df_numeric_ref_table['Var_Name'].unique():
                woes, bins, counts, targets, ivs = dc.woe(x_train[var].values, x_train[self.target].values,
                                                          auto=True, bin=tls.get_bin_range(df_numeric_ref_table, var))

                self.save_woe_png(woe_path, var, myfont)

        return x_train, x_test

    def save_woe_png(self, woe_path, var, myfont):
        plt.title(u"woe of {}".format(var), fontproperties=myfont)

        png_f = os.path.join(woe_path, "woe_{}.png".format(var))
        i = 0
        while os.path.exists(png_f):
            png_f = os.path.join(woe_path, "woe_{}_{}.png".format(var, i))
            i += 1

        plt.savefig(png_f, facecolor='w')
        plt.clf()

    @classmethod
    def woe_transform_by_reftable(cls, df, nominal_ref_table=None, numeric_ref_table=None):

        if nominal_ref_table is not None:
            # set nominal woe
            dc.set_nominal_woe(df, nominal_ref_table)

        if numeric_ref_table is not None:
            # set numeric woe
            woe_numeric_vars = numeric_ref_table.Var_Name.unique()
            for var in woe_numeric_vars:
                print("Setting woe for {} ...".format(var))
                df_ref_table_var = numeric_ref_table[numeric_ref_table['Var_Name'] == var]
                ref_table = dict(zip(df_ref_table_var['Var_Value'], df_ref_table_var['Ref_Value']))
                _ = dc.main_apply_numeric_ref_table(df, ref_table, var)

        out = df[[x for x in df.columns if x.endswith('woe')]]
        return out