import numpy as np
import pandas as pd

import PCProphet.io_ as io
import PCProphet.stats_ as st


def align(sample, shift):
    """
    receive a sample and shift it by shift by removing or adding fractions
    sample is mutable iteratable
    shift is a integer with sign
    """
    if shift > 0:
        sample = sample[:-shift]
        dummy = [0] * shift
        dummy.extend(sample)
        return dummy
    elif shift < 0:
        sample = sample[abs(shift) :]
        dummy = [0] * abs(shift)
        sample.extend(dummy)
        return sample
    else:
        return sample


def calc_shift(peak1, peak2):
    """
    for a dict of peaks return the average shift between all the peaks
    dict of peaks is complex -> protein -> peak
    """
    diff = []
    for cmplx in peak1:
        for prot in peak1[cmplx]:
            diff.append(peak1[cmplx][prot] - peak2[cmplx][prot])
    return round(st.medi(diff))


def split_align(x, shift, ids):
    if x["CREP"] == ids:
        aligned = align(x["INT"].split("#"), shift)
        pks = [int(float(pk)) for pk in x["PKS"].split("#")]
        if shift < 0:
            x["SEL"] = x["SEL"] - shift
            pks = [pk - shift for pk in pks]
        else:
            x["SEL"] = x["SEL"] + shift
            pks = [pk + shift for pk in pks]
        # if shift is less than 0 we are REMOVING let's put it back
        if x["SEL"] < 0:
            x["SEL"] = x["SEL"] + shift
            pks = [pk + shift for pk in pks]
        elif x["SEL"] > 72:
            x["SEL"] = x["SEL"] - shift
            pks = [pk - shift for pk in pks]
        x["INT"] = "#".join([str(x) for x in list(aligned)])
        x["PKS"] = "#".join([str(x) if x > 0 else 0 for x in list(pks)])
        return x["INT"]
    else:
        return x["INT"]


def align_samples(peaks_dict, sample, df):
    """
    pairwise loop of sample peaks
    """
    previous = "Ctrl1"
    # bckup old intensity column
    df["no_al"] = df["INT"].copy()

    align_id = {}
    # build lambda for realignement:
    align_id["Ctrl1"] = 0
    for repl in sample:
        if repl != "Ctrl1":
            print("Aligning {} to {}".format(repl, previous))
            shift = calc_shift(peaks_dict[repl], peaks_dict[previous])
            # we allign everything to Ctrl so previous is never touched
            df["INT"] = df.apply(lambda x: split_align(x, shift, repl), axis=1)
            align_id[repl] = shift
            previous = repl
            print("Detected shift of {} fraction for {}".format(str(shift), repl))
    return df, align_id


def housekeeping_complex(df, sample):
    """
    select complex sharing subunits between all experiments.
    filter peaks array
    """
    reps = lambda x, lookup: True if set(x["CREP"]) == set(lookup) else np.nan
    groups = ["CMPLX", "ID"]
    df2 = df.groupby(groups).apply(lambda row: reps(row, sample)).reset_index()
    if df2.empty:
        return pd.DataFrame()
    cm_f = lambda x: False if any(pd.isnull(x[0])) else True
    positive = df2.groupby("CMPLX").apply(lambda row: cm_f(row)).to_dict()
    positive = set({k: v for k, v in positive.items() if v}.keys())
    df = df[df["CMPLX"].isin(positive)]
    return df


def df2dict(df):
    """
    reformat df to have dict form
    cond => cmplx => prot
    """
    conds = io.makehash(io.makehash)
    for idx, row in df.iterrows():
        conds[row["CREP"]][row["CMPLX"]][row["ID"]] = row["SEL"]
    return dict(conds)


def qual_filter(df, p=0.5, cmplt=0.0):
    """
    use of high quality filter for complexes
    """
    mask = (df["P"] >= p) & (df["CMPLT"] >= cmplt)
    combfile = df[mask]
    return housekeeping_complex(combfile, set(combfile["CREP"]))


def runner(combfile_in, align_file, not_aligned):
    """
    read peaks file gets shared complexes between all conditions
    then loops pairwise and align each sample to previous
    so at the end all samples are aligned
    """
    combfile_b = pd.read_csv(combfile_in, sep="\t", index_col=False)
    combfile_b["CREP"] = combfile_b["COND"] + combfile_b["REPL"].map(str)
    # create backup
    combfile_b.to_csv(not_aligned, sep="\t", index=False)
    # if single replicate
    if len(set(combfile_b["CREP"])) == 1:
        combfile_b["no_al"] = combfile_b["INT"].copy()
        combfile_b.to_csv(combfile_in, sep="\t", index=False)
        return True
    common = qual_filter(combfile_b, 0.75, 0.75)
    if common.empty:
        print("Warning: High quality alignement fail\n")
        print("Global realignement will be done on all positive")
        common = qual_filter(combfile_b, 0.5, 0.0)
    # enforce simmilarity
    if common.empty or set(common["CREP"]) != set(combfile_b["CREP"]):
        print("No realignement performed")
        combfile_b["no_al"] = combfile_b["INT"].copy()
        combfile_b.to_csv(combfile_in, sep="\t", index=False)
        return True
    common.to_csv(align_file, sep="\t", index=False)
    refs = df2dict(common)
    aligned_df, shifts = align_samples(refs, set(combfile_b["CREP"]), combfile_b)
    aligned_df.to_csv(combfile_in, sep="\t", index=False)
    return True
