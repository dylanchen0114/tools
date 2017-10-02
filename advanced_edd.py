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


def row2fields(input_path, output_path, y_index, var_type=None, header=True):
    """
    
    split row in a vertical direction way, for low-memory computation
    ensure target y is in the last column
    
    :param input_path: input txt file path
    :param output_path: output each column's temp file ##directory## path
    :param y_index: target y index
    :param header: boolean, default True [must be True :)]
    :param var_type: a dict, key: column name, value: variable type (numeric, categorical)
    :return: write each column to output_path directory
    """

    num_fields = 0
    tmp_list = []
    keys = []

    # ensure the input file containing header
    if not header:
        raise OSError('Ensure header exists in your file !!')

    if not var_type:
        raise OSError('Must provide each Variables type !!')

    with open(input_path, 'r') as csv_file:
        data = csv.reader(csv_file, delimiter=',')

        for row_num, line in enumerate(data):

            if row_num == 0 and header:

                num_fields = len(line)
                keys = line
                print('Label Y Column Name: %s' % keys[y_index])

                for i in range(num_fields-1):

                    # create columns temp file and write header (one column, y label) to each file
                    tmp_list.append(open('%s/field' % output_path + str(i + 1) + '.tmp', 'w'))
                    tmp_list[i].write('{0},{1},{2}'.format(keys[i], 'variable_type', 'target_y') + "\n")

                continue

            # write each column together with y to the corresponding tmp file
            for i in range(num_fields-1):
                try:
                    tmp_list[i].write(','.join([line[i].replace(',', '，'), var_type[keys[i]], line[y_index]]) + "\n")
                except IndexError:
                    tmp_list[i].write("\n")

        for i in range(num_fields-1):
            tmp_list[i].close()


def feature_explore(output_path, input_file):

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

    with open(input_file) as file:

        # collect column x and target y into list
        for row_num, line in enumerate(file):
            x, var_type, y = line.strip().split(',')

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

    array_x = np.array(col_x)[value_mask].astype(np.float16)
    array_y = np.array(tar_y)[value_mask].astype(np.float16)

    for num, method in enumerate(type_algorithm[var_type]):
        if method == chi2:
            array_x = array_x.reshape(len(array_x), 1)
        statistic, p_value = method(array_x, array_y)
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


if __name__ == '__main__':

    root_directory = '/data/chendongyu/'
    variable_type_file_path = os.path.join(root_directory, 'var_type.csv')
    input_file_path = os.path.join(root_directory, 'implement_test.csv')
    output_file_path = os.path.join(root_directory, 'feature_explore.csv')

    print("Getting each variables' type ...")
    type_dict = {}
    with open(variable_type_file_path) as dict_file:
        for row_index, row_data in enumerate(dict_file):
            if row_index == 0:
                continue
            k, v = row_data.strip().split(',')
            type_dict.update({k: v})

    print("Splitting TXT file in a vertical direction way ...")
    row2fields(input_file_path, root_directory, y_index=-1, var_type=type_dict)

    file_list = glob.glob(os.path.join(root_directory, 'field*.tmp'))

    output_header = ['var_name', 'mean', 'std', 'No_obs', 'No_uniq', 'sum', 'missrate', 'misscnt', 'mod1', 'mod2', 'mod3',
                     'max', 'min', 'p1', 'p5', 'p10', 'p25', 'p33', 'p50', 'p66', 'p75', 'p90', 'p95', 'p99',
                     'statistic_0', 'p_value_0', 'statistic_1', 'p_value_1']

    print("Creating output header ...")
    with open(output_file_path, 'w') as header_file:
        header_file.write(','.join(output_header) + '\n')

    print("Computing each Column's descriptive statistics ... ")
    func = partial(feature_explore, output_file_path)
    pool = Pool(processes=10)
    p = pool.map(func, file_list)

    print("Removing temporary files ...")
    for remove_file in file_list:
        os.remove(remove_file)

    print('Done')
