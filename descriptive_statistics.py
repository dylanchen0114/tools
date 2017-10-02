# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""


num_list_first_letter = ['-', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']
num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'e', 'E', '+']


def check_float(string):
    len_str = len(string)
    float_ind = True
    for i in range(len_str):
        if i == 0:
            if string[i] not in num_list_first_letter:
                float_ind = False
                break
        else:
            if string[i] not in num_list:
                float_ind = False
                break
    return float_ind


def check_list_float(col_list):
    float_ind = True
    for item in col_list:
        if item == '':
            continue
        else:
            if not check_float(item):
                float_ind = False
            break
    return float_ind


def get_sorted(col_list):
    if not check_list_float(col_list):
        return []
    copy_list = []
    for item in col_list:
        if item != '':
            copy_list.append(float(item))

    copy_list.sort()
    return copy_list


def get_sum(sorted_list):
    if not sorted_list:
        return ''
    value = sum([float(item) for item in sorted_list])
    return value


def get_mod(col_list):
    freq = {}
    for i in range(len(col_list)):
        if col_list[i] != '':
            if col_list[i] in freq:
                freq[col_list[i]] += 1
            else:
                freq[col_list[i]] = 1
    max = 0
    mode = None
    for k, v in freq.items():
        if v > max:
            max = v
            mode = k
    max_1 = 0
    mode_1 = None
    for k, v in freq.items():
        if v > max_1 and k != mode:
            max_1 = v
            mode_1 = k
    max_2 = 0
    mode_2 = None
    for k, v in freq.items():
        if v > max_2 and k not in (mode, mode_1):
            max_2 = v
            mode_2 = k
    return [str(mode) + "::" + str(max), str(mode_1) + "::" + str(max_1), str(mode_2) + "::" + str(max_2)]


def get_max(sorted_list):
    if not sorted_list:
        return ''
    value = sorted_list[-1]
    return value


def get_min(sorted_list):
    if not sorted_list:
        return ''
    value = sorted_list[0]
    return value


def percentage(sorted_list, p):
    if not sorted_list:
        return ''
    percent = float(p) / 100
    index = max(1, int(float(len(sorted_list)) * percent)) - 1
    value = sorted_list[index]

    return value


def get_misscnt(col_list):
    value = 0
    for item in col_list:
        if item == '':
            value += 1
    return value


def get_std(mean, sorted_list):
    value = 0.0
    cnt_notmiss = 0
    if not sorted_list:
        return ''
    for item in sorted_list:
        value += (float(item) - mean) ** 2
        cnt_notmiss += 1
    if cnt_notmiss:
        value = round((value / cnt_notmiss) ** 0.5, 4)
    else:
        value = ''

    return value


def desc_stat(column_list):

    """
    h: {key1:{sum,mod_1,mod_2,mod_3,max,min,p1,p5,p10,p25,p50,p75,p90,p95,p99,missrate,misscnt,cnt,mean,std},key2:{},...} ->13 stats

    h only contains the variables in config_list(numeric variables)
    
    """

    output = {}

    cnt = len(column_list)
    cnt_uniq = len(set(column_list))
    misscnt = get_misscnt(column_list)
    missrate = round(float(misscnt) / cnt, 4)
    [mod, mode_1, mode_2] = get_mod(column_list)
    try:
        sorted_list = get_sorted(column_list)
        sum = get_sum(sorted_list)
        max = get_max(sorted_list)
        min = get_min(sorted_list)
        p1 = percentage(sorted_list, 1)
        p5 = percentage(sorted_list, 5)
        p10 = percentage(sorted_list, 10)
        p25 = percentage(sorted_list, 25)
        p33 = percentage(sorted_list, 33)
        p50 = percentage(sorted_list, 50)
        p66 = percentage(sorted_list, 66)
        p75 = percentage(sorted_list, 75)
        p90 = percentage(sorted_list, 90)
        p95 = percentage(sorted_list, 95)
        p99 = percentage(sorted_list, 99)
    except ValueError:
        sorted_list = []
        sum = ''
        max = ''
        min = ''
        p1 = ''
        p5 = ''
        p10 = ''
        p25 = ''
        p33 = ''
        p50 = ''
        p66 = ''
        p75 = ''
        p90 = ''
        p95 = ''
        p99 = ''

    if sum != '' and cnt - misscnt not in (0, ''):
        mean = round(float(sum) / (cnt - misscnt), 4)
    else:
        mean = ''
    std = get_std(mean, sorted_list)

    output = {"sum": sum, "mod1": mod, "mod2": mode_1, "mod3": mode_2, "max": max, "min": min,
              "p1": p1, "p5": p5, "p10": p10, "p25": p25, "p33": p33, "p50": p50, "p66": p66, "p75": p75,
              "p90": p90, "p95": p95, "p99": p99, "missrate": missrate, "misscnt": misscnt, "No_obs": cnt,
              "No_uniq": cnt_uniq, "mean": mean, "std": std}

    header = ['mean', 'std', 'No_obs', 'No_uniq', 'sum', 'missrate', 'misscnt', 'mod1', 'mod2', 'mod3',
              'max', 'min', 'p1', 'p5', 'p10', 'p25', 'p33', 'p50', 'p66', 'p75', 'p90', 'p95', 'p99']

    return output, header

