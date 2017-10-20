# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

To be continue (IV value)

"""

import csv
import glob
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.feature_selection import chi2

import descriptive_statistics


class AdvancedEdd:
    """

    split row in a vertical direction way, for low-memory computation
    ensure target y is in the last column

    header: boolean, default True [must be True :)]
    write each column to input_path directory

    """

    def __init__(self, input_path, var_type, output_path, y_index):

        self.input_path = input_path
        self.var_type = var_type
        self.output_path = output_path
        self.y_index = y_index

        pass

    def row2fields(self, var_type=None, header=True):

        num_fields = 0
        tmp_list = []
        keys = []

        # ensure the input file containing header
        if not header:
            raise OSError('Ensure header exists in your file !!')

        if not var_type:
            raise OSError('Must provide each Variables type !!')

        with open(self.input_path, 'r') as csv_file:
            data = csv.reader(csv_file, delimiter=',')

            for row_num, line in enumerate(data):

                if row_num == 0 and header:

                    num_fields = len(line)
                    keys = line
                    print('Label Y Column Name: %s' % keys[self.y_index])

                    for i in range(num_fields - 1):
                        # create columns temp file and write header (one column, y label) to each file
                        tmp_list.append(open('%s/field' % os.path.dirname(self.input_path) + str(i + 1) + '.tmp', 'w'))
                        tmp_list[i].write('{0},{1},{2}'.format(keys[i].replace(',', '，'), 'variable_type', 'target_y') + "\n")

                    continue

                # write each column together with y to the corresponding tmp file
                for i in range(num_fields - 1):
                    try:
                        tmp_list[i].write(
                            ','.join([line[i].replace(',', '，'), var_type[keys[i]], line[self.y_index]]) + "\n")
                    except IndexError:
                        tmp_list[i].write("\n")

            for i in range(num_fields - 1):
                tmp_list[i].close()

    def feature_explore(self, output_path, input_file):
        """
            Notes:
            1. ensure all missing values are represented by ''
            2. the statistic score is implemented without nan values
            3. this part is practised in parallel
            4. refer to function (descriptive_statistics) in edd.py

            """

        col_x = []
        tar_y = []
        key = ''
        var_type = ''

        type_algorithm = {'categorical': [chi2],
                          'numeric': [pearsonr]}

        with open(input_file) as csv_file:

            file = csv.reader(csv_file, delimiter=',')

            # collect column x and target y into list
            for row_num, line in enumerate(file):
                x, var_type, y = line

                if row_num == 0:
                    key = x
                    continue

                col_x.append(x)
                tar_y.append(float(y))
                var_type = var_type

        # get each variable type's corresponding statistic score
        des_dict, header = descriptive_statistics.desc_stat(col_x)

        # convert categorical columns to numeric
        if var_type == 'categorical':
            col_x = [col_x.index(x) if x != '' else '' for x in col_x]

        value_mask = [i for i, x in enumerate(col_x) if x != '']

        array_x = np.array(col_x)[value_mask].astype(np.float64)
        array_y = np.array(tar_y)[value_mask].astype(np.float64)

        for num, method in enumerate(type_algorithm[var_type]):
            if method == chi2:
                array_x = array_x.reshape(len(array_x), 1)

            try:
                statistic, p_value = method(array_x, array_y)
            except ValueError:
                statistic = ' '
                p_value = ' '

            if method == chi2:
                statistic = statistic[0]
                p_value = p_value[0]
            des_dict.update({'statistic_%s' % num: statistic, 'p_value_%s' % num: p_value})
            header += ['statistic_%s' % num, 'p_value_%s' % num]

        header = ['var_name'] + header
        des_dict.update({'var_name': key})

        output = []
        with open(output_path, 'a') as out_file:
            for name in header:
                output.append(str(des_dict[name]))
            out_file.write(','.join(output) + '\n')

    def run(self):

        print("Getting each variables' type ...")
        type_dict = {}
        with open(self.var_type) as csv_file:

            dict_file = csv.reader(csv_file, delimiter=',')

            for row_index, row_data in enumerate(dict_file):
                if row_index == 0:
                    continue
                k, v = row_data
                type_dict.update({k: v})

        print("Splitting TXT file in a vertical direction way ...")
        self.row2fields(var_type=type_dict)

        output_header = ['var_name', 'mean', 'std', 'No_obs', 'No_uniq', 'sum', 'missrate', 'misscnt', 'mod1', 'mod2',
                         'mod3', 'max', 'min', 'p1', 'p5', 'p10', 'p25', 'p33', 'p50', 'p66', 'p75', 'p90', 'p95',
                         'p99',
                         'statistic_0', 'p_value_0', 'statistic_1', 'p_value_1']

        print("Creating output header ...")
        with open(self.output_path, 'w') as header_file:
            header_file.write(','.join(output_header) + '\n')

        file_list = glob.glob(os.path.join(os.path.dirname(self.input_path), 'field*.tmp'))

        print("Computing each Column's descriptive statistics ... ")
        func = partial(self.feature_explore, self.output_path)
        pool = Pool(processes=10)
        p = pool.map(func, file_list)

        print("Removing temporary files ...")
        for remove_file in file_list:
            os.remove(remove_file)

        print('Done')
