# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""


import pandas as pd


def xlsx2df(file_name, sheet_index):

    xl = pd.ExcelFile(file_name)
    print('Sheet Names in File: ', xl.sheet_names)

    sheet_file = xl.sheet_names[sheet_index]
    df = xl.parse(sheet_file)

    return df
