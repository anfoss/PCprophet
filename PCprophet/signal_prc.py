import numpy as np
import scipy.ndimage as image
import scipy.signal as signal_processing


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
