# coding=utf-8
import os

from pyspark import SparkContext, RDD
from operator import itemgetter
import numpy as np

if __name__ == '__main__':
    pass


def collect_spark_output(output_dir, target_file):
    os.system('cat %s/part* > %s' % (output_dir, target_file))
    os.system('rm -r %s' % output_dir)


def spark_output(rdd, out_file):
    rdd.saveAsTextFile("file:///%s.out" % out_file)
    collect_spark_output(out_file+".out", out_file)


def spark_readlines(filename, sc, repart=True):
    assert isinstance(sc, SparkContext)
    lines = sc.textFile("file:///"+filename)
    if repart: lines = lines.repartition(sc.defaultParallelism*3)
    return lines


def spark_read_data(filename, splitter, sc, repart=True):
    lines = spark_readlines(filename, sc, repart)
    data = lines.map(fc_extract_line(splitter))
    return data


def plus_str(data):
    return "+".join(data)


def tuple_split(i):
    """
    :param i: split data into tuple A,B, i is A's length
    :return: the splitter function
    """
    def spliter(data):
        return tuple(data[:i]), tuple(data[i:])
    return spliter



def tuple_merge(d):
    j, k = d
    return tuple(j) + tuple(k)

def ab_c__a_bc(data):
    (a,b), c = data
    return a, (b, c)

def a_bc__ab_c(data):
    a, (b,c) = data
    return (a,b), c

def a_b__ab_b(data):
    a, b = data
    return (a,b), b

def _tuple_(item):
    if type(item) in (str, unicode):
        return (item,)
    try:
        t = tuple(item)
    except:
        t = (item,)
    return t

def tuple_flat(d):
    d1, d2 = d
    return _tuple_(d1) + _tuple_(d2)


def mp_mapping(mapping):
    def _map_(data):
        return tuple(mapping[i](data[i]) for i in xrange(len(data)) if mapping[i])
    return _map_


def mp_keyValue_(d):
    return d[0], d[1:]



def sort_(data):
    k,li = data
    return k, sorted(li, key=itemgetter(2))

def fc_sort(i):
    def _sort(data):
        k, li = data
        return k, sorted(li, key=itemgetter(i))
    return _sort

def extract_line(line, splitter='\t'):
    return line.strip().split(splitter)


def fc_extract_line(splitter):
    def extractor_(line):
        return line.strip().split(splitter)
    return extractor_


def mp_print_line(items, splitter='\t'):
    return splitter.join(("%s"%x for x in items))


def fc_print_line(splitter):
    def _print_line(items):
        return splitter.join("%s"%x for x in items)
    return _print_line


def partitionBy(rdd, f):
    rdd_cached = rdd.cache()
    passes = rdd_cached.filter(f)
    fails = rdd_cached.filter(lambda d: not f(d))
    return passes, fails