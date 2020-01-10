import math as m
import statistics as stat
import numpy as np
import scipy.ndimage as image
import scipy.signal as signal_processing


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


# fast combinations
def fast_comb(a, _=None):
    n = len(a)
    L = n * (n - 1) // 2
    iterID = 0
    for i in range(n):
        for j in range(i + 1, n):
            iterID += 1
            yield a[i], a[j]


# signal proc
def gauss_filter(arr, sigma=1, order=0):
    """
    perform gaussian filtering on the data
    """
    filt = image.filters.gaussian_filter1d(arr, sigma=sigma, order=order)
    return filt


def peak_picking(arr, height=0.1, width=2):
    """
    return indexes of peaks from array giving a peak of minimum height var
    """
    peaks = signal_processing.find_peaks(
        arr, height=height, width=width
    )
    return peaks


def resize(ls, lower=0, upper=1.0):
    """
    rescale list of values from 1 to 0
    """
    if max(ls) == min(ls):
        return [0] * len(ls)
    else:
        ls_std = [(x - min(ls)) / (max(ls) - min(ls)) for x in ls]
        return [(x * (upper - lower) + lower) for x in ls_std]


def resample(signal_l, input_fr, output_fr):
    """
    use linear interpolation
    using endpoint=False gets less noise in the resampled
    """
    scale = output_fr / input_fr
    n = round(len(signal_l) * scale)
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),
        np.linspace(0.0, 1.0, len(signal_l), endpoint=False),
        signal_l,
    )
    return resampled_signal


def impute_namean(ls):
    """
    impute 0s in list with value in between if neighbours are values
    assumption is if data is gaussian mean of sequential points is best
    """
    idx = [i for i, j in enumerate(ls) if j == 0]
    for zr in idx:
        if zr == 0 or zr == (len(ls) - 1):
            continue
        elif ls[zr - 1] != 0 and ls[zr + 1] != 0:
            ls[zr] = (ls[zr - 1] + ls[zr + 1]) / 2
        else:
            continue
    return ls


def fwhm(y, frac=2):
    """
    calculate full width half max of peak within two fractions
    """
    if not y:
        return np.nan
    y = np.array(y)
    x = [x for x in range(1, (len(y) + 1))]
    d = y - (max(y) / frac)
    indexes = np.where(d > 0)[0]
    try:
        return abs(x[indexes[-1]] - x[indexes[0]])
    except IndexError as e:
        return np.nan


def resize_plot(arr, input_fr, output_fr):
    """
    used only for plots, to change in next versions
    """
    if input_fr == output_fr:
        return [float(x) for x in arr.split("#")]
    tmp = [float(x) for x in arr.split("#")]
    return resize(resample(np.array(tmp), int(input_fr), int(output_fr)))
