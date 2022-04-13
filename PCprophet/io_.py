import re
import pandas as pd
import numpy as np
import sys
import os
import networkx as nx
from datetime import datetime
from collections import defaultdict
import random
import time
import uuid


def makehash(w=dict):
    """autovivification like hash in perl
    http://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python
    use call it on hash like h = makehash()
    then directly
    h[1][2]= 3
    useful ONLY for a 2 level hash
    """
    # return defaultdict(makehash)
    return defaultdict(w)


def makedeephash():
    """autovivification like hash in perl
    http://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python
    use call it on hash like h = makehash()
    then directly
    h[1][2]= 3
    useful ONLY for a 2 level hash
    """
    # return defaultdict(makehash)
    return defaultdict(makedeephash)


def makehashlist():
    """autovivification like hash in perl
    http://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python
    use call it on hash like h = makehash()
    then directly
    h[1][2]= 3
    useful ONLY for a 2 level hash
    """
    # return defaultdict(makehash)
    return defaultdict(list)


def makehashset():
    """autovivification like hash in perl
    http://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python
    use call it on hash like h = makehash()
    then directly
    h[1][2]= 3
    useful ONLY for a 2 level hash
    """
    # return defaultdict(makehash)
    return defaultdict(set)


def reformat_dict(dic, is_list=False):
    """
    get dict with delim and things and returns two object
    1) nr => fullname
    2) nr = value
    """
    nr2fullname = {}
    nr2feat = {}
    for k, v in dic.items():
        k = k.strip('"')
        if is_list:
            v = [x.replace('"', "") for x in v]
        else:
            v = str(v).strip('"')
        n = k.split("_")[-1]
        nr2feat[n] = v
        nr2fullname[n] = k
    return nr2feat, nr2fullname


def df2dict(path, k, v):
    tmp = pd.read_csv(path, sep="\t")
    return dict(zip(list(tmp[k]), list(tmp[v])))


def create_unique(dic):
    """
    create unique identifier from the dict
    """
    return dict(zip(range(1, len(dic.keys()) + 1), dic.keys()))


def reformat_dict_f(dic, mapping):
    """
    switch keys with key values (unique identifier)
    """
    return {k: dic[v] for k, v in mapping.items()}


def read_sample_ids_diff(info_path):
    """
    read sample ids and return a hash
    cond => short ID
    """
    header = []
    HoH = {}
    temp = {}
    for line in open(info_path, "r"):
        line = line.rstrip("\n")
        if line.startswith(str("Sample") + "\t"):
            header = re.split(r"\t+", line)
        else:
            things = re.split(r"\t+", line)
            temp = dict(zip(header, things))
        if temp:
            HoH[temp["cond"]] = temp["short_id"]
    return HoH


def create_df(prot_dict):
    df = pd.DataFrame.from_dict(prot_dict)
    df = df.T
    # df.drop_duplicates(subset=None, keep='first', inplace=True)
    df.fillna(value=0, inplace=True)
    df[(df.T != 0).all()]
    return df


def create_file(filename, header):
    """
    create file in filename
    header is list
    """
    with open(filename, "w", encoding="utf-8") as outfile:
        outfile.write("%s\n" % "\t".join([str(x) for x in header]))


def dump_file(filename, things):
    """
    dump things to file to filename
    """
    with open(filename, "a", encoding="utf-8") as outfile:
        outfile.write("%s\n" % things)


def read_pred(pred_path):
    """
    collapse prediction into protein groups
    need to modify prediction to add complex member and also protein names
    """
    header = []
    temp = {}
    test = {}
    for line in open(pred_path, "r"):
        line = line.rstrip("\n")
        if line.startswith(str("ID") + "\t"):
            header = re.split(r"\t+", line)
        else:
            things = re.split(r"\t+", line)
            temp = dict(zip(header, things))
            # # TODO deal with duplicate entries in database
            test[temp["ID"]] = float(temp["POS"])
    return test


def read_mp_feat(pred_path):
    """
    if no prediction was done take mp_feat_norm
    """
    header = []
    temp = {}
    test = makehash()
    for line in open(pred_path, "r"):
        line = line.rstrip("\n")
        if line.startswith(str("ID") + "\t"):
            header = re.split(r"\t+", line)
        else:
            things = re.split(r"\t+", line)
            # need to deal with error being a thing or empty col
            temp = dict(zip(header, things))
            test[temp["ID"]] = []
            test[temp["ID"]].extend(temp["MB"].split("#"))
            # test[temp['ID']][temp['MB']] = 'yes'
    return test


def read_matrix(path, arr=False):
    """
    read matrix and returns HoA[protein] = # delim int
    """
    header = []
    HoA = makehash()
    temp = {}
    for line in open(path, "r"):
        line = line.rstrip("\n")
        if line.startswith(str("ID") + "\t"):
            header = re.split(r"\t+", line)
        else:
            things = re.split(r"\t+", line)
            temp = dict(zip(header, things))
        if temp:
            val = "#".join([temp[key] for key in header if key != "ID"])
            HoA[temp["ID"]] = val
    return HoA


def read_peaks(path, arr=False):
    """
    read peak list in the form of
    prot  peaks selected cmplx name
    output hash cmplx name prot => peaks selected
    """
    header = []
    HoA = makehash()
    temp = {}
    for line in open(path, "r"):
        line = line.rstrip("\n")
        if line.startswith("MB\tID"):
            header = re.split(r"\t+", line)
        else:
            things = re.split(r"\t+", line)
            temp = dict(zip(header, things))
        if temp:
            row = "\t".join([temp["PKS"], temp["SEL"]])
            HoA[temp["ID"]][temp["MB"]] = row
    return HoA


def read_sample_ids(info_path):
    """
    read sample to treatment
    """
    header = []
    HoH = {}
    temp = {}
    for line in open(info_path, "r"):
        line = line.rstrip("\n")
        if line.startswith(str("Sample") + "\t"):
            header = re.split(r"\t+", line)
        else:
            things = re.split(r"\t+", line)
            temp = dict(zip(header, things))
        if temp:
            HoH[temp["Sample"]] = "_".join([temp["cond"], temp["repl"]])
    return HoH


def read_txt(path, first_col="GN"):
    """
    read a tab delimited file giving a path and the first column name
    return a hash of hashes prot => sample => val
    """
    header = []
    HoA = makehash()
    temp = {}
    for line in open(path, "r"):
        line = line.rstrip("\n")
        if line.startswith(str(first_col) + "\t"):
            header = re.split(r"\t+", line)
        else:
            things = re.split(r"\t+", line)
            temp = dict(zip(header, things))
        if temp:
            HoA[temp.get("GN")] = []
            for key in header:
                try:
                    HoA[temp.get("GN")].append(float(temp[key]))
                except ValueError as v:
                    continue
    return HoA


def ppi2graph(infile):
    df = pd.read_csv(infile, sep="\t")
    ppi = dict(zip(df["protA"], df["protB"]))
    n = nx.Graph()
    for k in ppi.keys():
        n.add_edge(k, ppi[k])
    return n


def wrout(d, filename, header, is_hyp=False):
    """
    giving a list, a filename and a set of headers (tab delimited)
    """
    with open(filename, "w", encoding="utf-8") as outfile:
        outfile.write("\t".join(header) + "\n")
        for k in d:
            if is_hyp:
                base_id = uniqueid()
                cmplx_nr = "cmplx_" + str(base_id)
                line = "\t".join([cmplx_nr, k, str(d[k])])
                outfile.write(str(line) + "\n")
            else:
                outfile.write(str(k) + "\n")
    # print('file saved in ' + str(filename))
    return True


def read_combined(combfile):
    """
    receive a combined file and uniforms the annotation
    """
    HoA = makehashlist()
    df = pd.read_csv(combfile, sep="\t")
    for index, row in df.iterrows():
        HoA[row["CMPLX"]].append(row["ID"])
    return HoA


def create_db_from_cluster(nodes, clusters):
    idx = 1
    ids = "ppi"
    header = ["ComplexID", "ComplexName", "subunits(Gene name)"]
    path = resource_path("./ppi_db.txt")
    create_file(path, header)
    for cmplx in clusters:
        nm = ";".join([str(nodes[x]) for x in list(cmplx)])
        tmp = "_".join([ids, str(idx)])
        dump_file(path, "\t".join([str(idx), tmp, nm]))
        idx += 1
    return True


def file2folder(file_, prefix="./tmp/"):
    # we are already stripping the extension
    filename = os.path.splitext(os.path.basename(file_))[0]
    return os.path.join(prefix, filename)


def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller
    """
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def catch(func, handle=lambda e: e, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle(e)


def split_hypo_db(pred, ref_cmplx, cmplx_ann):
    """
    read prediction ref cmplx and cmplx and return the positive
    """
    pred = pd.read_csv(pred, sep="\t", index_col=False)
    ref_cmplx = pd.read_csv(ref_cmplx, sep="\t")
    ann = pd.read_csv(cmplx_ann, sep="\t")
    on = ["ID"]
    xx = lambda x, y, on: pd.merge(x, y, how="left", left_on=on, right_on=on)
    mrg = xx(ann, xx(pred, ref_cmplx, on), ["ID"])
    pos = mrg["IS_CMPLX"] == "Yes"
    mrg = mrg[pos]
    # split in hypothesis and db and then return
    hyp = mrg[(mrg.ANN == 0)]
    db = mrg[mrg.ANN == 1]
    return hyp, db


def uniqueid():
    """
    generate unique id with length 17 to 21
    """
    return str(uuid.uuid4())


def split_to_df(df, col, sep=","):
    tmp = pd.DataFrame(df[col].str.split(sep).tolist(), index=df.index.copy())
    return tmp


def explode(df, lst_cols, fill_value="", preserve_index=False):
    # make sure `lst_cols` is list-alike
    """
    https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows/40449726#40449726
    """
    if (
        lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))
    ):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = pd.DataFrame(
        {col: np.repeat(df[col].values, lens) for col in idx_cols}, index=idx
    ).assign(**{col: np.concatenate(df.loc[lens > 0, col].values) for col in lst_cols})
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = res.append(df.loc[lens == 0, idx_cols], sort=False).fillna(fill_value)
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res


def prepare_feat(infile, thresh=1, missing=["nan", "na", "", None, "n", "-"]):
    """
    read infile and split it
    """
    feat = pd.read_csv(infile, sep="\t", na_values=missing)
    feat.dropna(inplace=True)
    memos = feat[["ID"]]
    torm = ["ID", "MB", "SC_CC", "SC_MF", "SC_BP", "TOTS"]
    feat.drop(torm, axis=1, inplace=True)
    cor = split_to_df(feat, "COR")
    dif = split_to_df(feat, "DIF")
    feat2 = feat[["SHFT", "W"]]
    feat2 = feat2.apply(pd.to_numeric)
    feat_num = pd.concat([feat2, cor, dif], axis=1)
    feat_num.replace(to_replace=missing, value=np.nan, inplace=True)
    # remove nan
    mask = feat_num.isnull().mean(axis=1) <= thresh
    feat_num = feat_num[mask].fillna(0)
    return feat_num.values, memos[mask].values


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed
