import sys
import re
import os
import numpy as np
import pandas as pd
from scipy import cluster

import PCProphet.signal_prc as preproc
import PCProphet.io_ as io


# standardize and center methods
def center_arr(hoa, fr_nr="all", norm=True, nat=True, stretch=(True, 72)):
    hypo = {}
    for k in hoa:
        key = hoa[k]
        if fr_nr != "all":
            key = key[0:(fr_nr)]
        if len([x for x in key if x > 0]) < 2:
            continue
        key = preproc.gauss_filter(key, sigma=1, order=0)
        key = preproc.impute_namean(key)
        if stretch[0]:
            # input original length wanted length
            key = preproc.resample(key, len(key), output_fr=stretch[1])
        key = preproc.resize(key)
        hypo[k] = list(key)
    return hypo


def split_peaks(prot_arr, pr, skp=0):
    """
    split peaks in n samples giving skp fractions of window
    returns
    'right_bases': array([32]), 'left_bases': array([7])
    """
    peaks = list(preproc.peak_picking(prot_arr, 0.2, width=4))
    left_bases = peaks[1]["left_bases"]
    right_bases = peaks[1]["right_bases"]
    fr_peak = peaks[0]
    ret = {}
    # if no return value or 1 peak
    if len(fr_peak) < 2:
        ret[pr] = prot_arr
        return ret
    for idx, pk in enumerate(fr_peak):
        if pk < 6 and pk > 69:
            continue
        nm = "_".join([pr, str(idx)])
        clean = fill_zeroes(prot_arr, pk, left_bases[idx], right_bases[idx])
        ret[nm] = clean
    return ret


def fill_zeroes(prot, pk, left_base, right_base):
    """
    check left and right side of peaks and zero if >
    """
    arr = prot.copy()
    arr[:left_base] = [0 for aa in arr[:left_base]]
    arr[right_base:] = [0 for aa in arr[right_base:]]
    right = zero_sequence(arr[pk : len(arr)])
    left = zero_sequence(arr[:pk][::-1])[::-1]
    return left + right


def zero_sequence(arr):
    idx = 0
    k = True
    while k:
        # if we are at end return array
        if idx == len(arr) - 1:
            return arr
        # if current value smaller than next (i.e increasing)
        elif arr[idx] < arr[(idx + 1)]:
            # slice until there
            tmp = arr[:idx]
            l = [0] * (len(arr) - len(tmp))
            return tmp + l
        idx += 1


def decondense(df, ids):
    """
    decondense a linkage matrix into all flat clusters
    """
    clusters = {}
    rows = cluster.hierarchy.linkage(df)
    lab = dict(zip(range(len(ids) + 1), ids))
    for row in range(rows.shape[0]):
        cluster_n = row + len(ids)
        glob1, glob2 = rows[row, 0], rows[row, 1]
        current = []
        for glob in [glob1, glob2]:
            if glob > (len(ids) - 1):
                current += clusters[glob]
            else:
                current.append(lab[int(glob)])
        clusters[cluster_n] = current
    return clusters


def format_cluster(hoa, clust, max_hypothesis):
    out = {}
    lk = {k: ",".join(map(str, v)) for k, v in hoa.items()}
    for gn in clust.values():
        if len(gn) > 1 and len(gn) <= max_hypothesis:
            gn = [x if x in lk else re.sub("_\d+$", "", x) for x in gn]
            out["#".join(gn)] = "#".join([lk[x] for x in gn])
    return out


def collapse_prot(infile, max_hypothesis, use):
    prot = io.read_txt(infile, "GN")
    prot = center_arr(prot, fr_nr=use, stretch=(True, 72))
    prot2 = {}
    for pr in prot:
        pks = split_peaks(prot[pr], pr)
        if pks:
            for k in pks:
                prot2[k] = pks[k]
    pr_df = io.create_df(prot2)
    z = decondense(pr_df, list(pr_df.index))
    hypothesis = format_cluster(prot, z, max_hypothesis)
    #  return peaks2prot(hypothesis, prot),pr_df
    return hypothesis, pr_df


def runner(infile, hypothesis, max_hypothesis, use_fr):
    """
    generate hypothesis from infile using all fract fractions and max hypo nr
    """
    if hypothesis is "all":
        print("Generating hypothesis for " + infile)
        hypo, df_s = collapse_prot(
            infile=infile, max_hypothesis=int(max_hypothesis), use=use_fr
        )
        base = io.file2folder(infile, prefix="./tmp/")
        nm = os.path.join(base, "hypo.txt")
        io.wrout(hypo, nm, ["ID", "MB", "FT"], is_hyp=True)
        df_s.to_csv(os.path.join(base, "splitted_transf.txt"), sep="\t")
    else:
        pass


if __name__ == "__main__":
    main()
