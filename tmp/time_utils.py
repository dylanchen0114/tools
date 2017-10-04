# -*- coding: utf-8 -*-


import datetime


def _timestamp():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M")
    return now_str


def _timestamp_pretty():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M")
    return now_str

def stamp_to_str(time_stamp):
    """将 13 位整数的毫秒时间戳转化成本地普通时间 (字符串格式)
    :param timestamp: 13 位整数的毫秒时间戳 (1456402864242)
    :return: 返回字符串格式 {str}'2016-02-25 20:21:04.242000'
    """
    local_str_time = datetime.datetime.fromtimestamp(float(time_stamp) / 1000.0).strftime('%Y-%m-%d %H:%M:%S.%f')
    return local_str_time
