import math as m
import statistics as stat
import numpy as np


# basic stat
def mean(numbers, as_decimal=False):
    """
    compute mean
    """
    if sum([float(x) for x in numbers]) > 0:
        m = (sum(numbers)) / max(len(numbers), 1)
        return m
    else:
        return 0


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def renormalize(n, range1, range2):
    """
    normalize a n value in a new range from the old range
    """
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]


def medi(numbers):
    """
    compute median
    """
    return stat.median(numbers)


def overl(l1, l2):
    """
    calculate overlap of two lists
    """
    return len(set(l1).intersection(l2)) / max([len(l1), len(l2)])


def percentile(nrs, percent, key=lambda x: x):
    """
    Find the percentile of a list of values.
    nrs is input list (need to be sorted)
    percent is a float value from 0.0 to 1.0.
    key optional to compute value from each element of N.
    return nr percentile
    """
    k = (len(nrs) - 1) * percent
    f = m.floor(k)
    c = m.ceil(k)
    if f == c:
        return key(nrs[int(k)])
    d0 = key(nrs[int(f)]) * (c - k)
    d1 = key(nrs[int(c)]) * (k - f)
    return d0 + d1


# rescale
def scale(val, fi, se, factor):
    """
    rescale value between 1 and 0 giving two bounds
    """
    des = ((val - se) / (fi - se)) ** factor
    return des


# desiderability
def desi_xtrm(val, fi, se, th, fo, scl=2):
    """
    map 1 if val lt fi or gt fo, 0 if between se && th,
    scale if in the middle. 2 different windows for asymmetric distributions
    """
    if val <= fi or val >= fo:
        return 1
    elif val >= se and val <= th:
        return 0
    elif val > fi and val < se:
        # using a curve instead of a line for the scaling
        score = scale(val, fi, se, scl)
        return score
    elif val > th and val < fo:
        score = scale(val, fo, th, scl)
        return score
    elif val is None or m.isnan(val) is True:
        return 0


def desi_mt_2windows(val, fi, th, scl=2):
    """
    map 1 if val mt fi, 0 if val lt th, scale if middle.
    """
    if val < fi:
        return 0
    elif val >= th:
        return 1
    elif val >= fi and val < th:
        score = scale(val, th, fi, scl)
        return score
    elif val is None or m.isnan(val) is True:
        return 0


def desi_lt_2windows(val, fi, se, scl=2):
    """
    map 1 if val gt th same as lt
    """
    if val <= fi:
        return 1
    elif val > se:
        return 0
    elif val > fi and val <= se:
        score = scale(val, fi, se, scl)
        return score
    elif val is None or m.isnan(val) is True:
        return 0


def aggr_de(desi_dict, desi_weight):
    """
    read a dictionary of desiderability
    and a dict of weight and return the geometric average
    """
    tmp = 0
    sum_w = sum(list(w.values()))
    if sum(list(ds.values())) == 0:
        return 0
    for key in ds:
        if ds[key] != 0:
            d = w[key] * m.log(ds[key]) / sum_w
            tmp += d
    dummy = list(ds.values())
    dummy.append(m.exp(tmp))
    if debug:
        io.dump_file(".debug_desi.txt", "\t".join([str(x) for x in dummy]))
    return m.exp(tmp)


# fast combinations
def fast_comb(a, _=None):
    n = len(a)
    L = n * (n - 1) // 2
    iterID = 0
    for i in range(n):
        for j in range(i + 1, n):
            iterID += 1
            yield a[i], a[j]
