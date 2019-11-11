import re
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
import random as random

import PCProphet.io_ as io
import PCProphet.stats_ as st


def db2ppi(list_sep):
    """
    read dbfile and convert complexes to ppi for faster quering
    """
    ppi_db = nx.Graph()
    for members in list_sep:
        for pairs in combinations(members.split("#"), 2):
            ppi_db.add_edge(str.upper(pairs[0]), str.upper(pairs[1]))
    ppi_db.remove_edges_from(ppi_db.selfloop_edges())
    return ppi_db


def get_fdr(tp, fp, tn, fn):
    if fp == 0:
        return 0
    return fp / (tp + fp)


def get_ppv(tp, fp, tn, fn):
    return tp / (tp + fp)


def get_tpr(tp, fp, tn, fn):
    return tp / (tp + fn)


def overlap_net(ppi_network, mb, over=0.5):
    """
    calculates the overlap between a network and a list
    """
    match, nomatch = 0, 0
    mb = re.split(r"#", mb)
    if len(mb) < 2:
        return True
    for pairs in combinations(mb, 2):
        if ppi_network.has_edge(pairs[0], pairs[1]):
            match += 1
        else:
            nomatch += 1
    if match / (nomatch + match) >= over:
        return True
    else:
        return False


#
def calc_fdr(hypo, db, go_thresh):
    """
    calculate confusion matrix from test and db
    db is a networkX object
    """
    test = dict(zip(list(hypo["TOTS"]), list(hypo["MB_x"])))
    isindb = {k: overlap_net(db, v) for k, v in test.items()}
    est_fdr = []
    for go_score in go_thresh:
        tp, fp, tn, fn = 0, 0, 0, 0
        for cm in test:
            cm = float(cm)
            if cm >= go_score and isindb[cm]:
                tp += 1
            elif cm >= go_score:
                fp += 1
            elif cm < go_score and isindb[cm]:
                fn += 1
            elif cm < go_score:
                tn += 1
        est_fdr.append(get_fdr(tp, fp, tn, fn))
        # print(tp, fp, tn, fn, get_fdr(tp, fp, tn, fn))
    return est_fdr


def estimate_cutoff(fdr_arr, db_thresh, target_fdr=0.5):
    """
    estimate corum cutoff for target FDR
    use the lowest percentage of corum for reaching target fdr
    """
    fdr2thresh = dict(zip(fdr_arr, db_thresh))
    fdr_min = min([abs(x - target_fdr) for x in fdr_arr])
    idx = [abs(x - target_fdr) for x in fdr_arr].index(fdr_min)
    return fdr2thresh[fdr_arr[idx]]


def loop_db(db):
    """
    loop through sumGO for corum and estimate fdr
    """
    db_thresh = []
    counter = 1
    while counter <= 100:
        db_thresh.append(st.percentile(db.values, counter / 100))
        counter += 1
    return db_thresh


def filter_hypo(hypo, db, estimated_fdr):
    """
    return object for collapse py
    """
    mask = (hypo["ANN"] == 0) & (hypo["TOTS"] >= estimated_fdr)
    filt = db.append(hypo[mask], ignore_index=True)
    # now we reformat out to work with collapse.dedup
    hyp_out, cor_out = io.makehash(), io.makehash()
    for k in filt["ID"]:
        v = filt.MB_y.values[filt["ID"] == k][0].split(",")
        if filt.ANN.values[filt["ID"] == k][0] == 0:
            hyp_out[k]["S"] = filt.TOTS.values[filt["ID"] == k][0]
            hyp_out[k]["V"] = []
            hyp_out[k]["V"].extend(v)
        else:
            cor_out[k] = []
            cor_out[k].extend(v)
    return hyp_out, cor_out


def fdr_from_GO(pred, db, cmplx_ann, target_fdr, fdrfile):
    """
    use positive predicted annotated from db to estimate hypothesis fdr
    """
    hypo, db = io.split_hypo_db(pred, db, cmplx_ann)
    io.create_file(fdrfile, ["fdr", "sumGO"])
    if hypo.empty or len(set(hypo["TOTS"])) == 1:
        return filter_hypo(hypo, db, 0)
    estimated_fdr = 0.5
    if target_fdr > 0:
        ppi_db = db2ppi(db["MB_x"])
        try:
            db_thresh = loop_db(db["TOTS"])
        # if not enough positive we used prefixed threshold of .5
        except IndexError as e:
            db_thresh = [0.5]
        thresh_fdr = calc_fdr(hypo, ppi_db, db_thresh)
        estimated_fdr = estimate_cutoff(thresh_fdr, db_thresh, target_fdr)
        for pairs in zip(thresh_fdr, db_thresh):
            io.dump_file(fdrfile, "\t".join(map(str, pairs)))
        print("Estimated GO cutoff is for {} is {}".format(pred, estimated_fdr))
        return filter_hypo(hypo, db, estimated_fdr)
    else:
        return filter_hypo(hypo, db, 0)
