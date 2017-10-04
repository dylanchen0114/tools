# -*- coding: utf-8 -*-

import json

def csv2dicts(csvfile, names=None):
    """
    turn csv file into dict
    :param csvfile: iterator from csv.reader
    :param names: list of column names
    :return: dict, header as key, contents as value
    """
    data = []
    for row_index, row in enumerate(csvfile):
        if row_index == 0:
            if names:
                keys = names
            else:
                keys = row
                print(keys)
            continue
        data.append({key: value for key, value in zip(keys, row)})
    return data


def json2dicts(data, json_column):
    """
    update json dict to existing dict

    """
    for i, x in enumerate(data):
        for key, value in x.items():
            if key == json_column and x[key]:
                jd = json.loads(value)
        data[i].update(jd)
        data[i].pop(json_column)
        if 'NULL' in x:
            x.pop('NULL')


def set_nan_as_string(data, replace_str='0'):
    """
    replace csv dict nan values
    :param data: csv dict
    :param replace_str: 
    :return: replaced csv dict
    """
    for i, x in enumerate(data):
        for key, value in x.items():
            if value == '':
                x[key] = replace_str
        data[i] = x




