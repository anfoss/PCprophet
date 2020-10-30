import re
import numpy as np
import pandas as pd
import networkx as nx
import random as random
from sklearn.mixture import GaussianMixture

import PCprophet.io_ as io
import PCprophet.stats_ as st


def db2ppi(list_sep):
    """
    read dbfile and convert complexes to ppi for faster quering
    """
    ppi_db = nx.Graph()
    for members in list_sep:
        for pairs in st.fast_comb(members.split("#"), 2):
            ppi_db.add_edge(str.upper(pairs[0]), str.upper(pairs[1]))
    ppi_db.remove_edges_from(nx.selfloop_edges(ppi_db, keys=True))
    return ppi_db


def get_fdr(tp, fp, tn, fn):
    if fp == 0:
        return 0
    return fp / (tp + fp)


def overlap_net(ppi_network, mb, over=0.5):
    """
    calculates the overlap between a network and a list
    """
    match, nomatch = 0, 0
    mb = re.split(r"#", mb)
    if len(mb) < 2:
        return True
    for pairs in st.fast_comb(mb, 2):
        if ppi_network.has_edge(pairs[0], pairs[1]):
            match += 1
        else:
            nomatch += 1
    if match / (nomatch + match) >= over:
        return True
    else:
        return False


def calc_fdr(combined, db, go_thresh):
    """
    calculate confusion matrix from test and db
    db is a networkX object
    """
    test = dict(zip(list(combined["TOTS"]), list(combined["MB"])))
    isindb = {k: overlap_net(db, v) for k, v in test.items()}
    est_fdr = []
    conf_m = []
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
        conf_m.append("\t".join(map(str, [tp, fp, tn, fn])))
    return est_fdr, conf_m


def calc_pdf(decoy, target=None):
    """
    calculate empirical fdr if not enough db hits are present
    use gaussian mixture model 2 components to predict class probability
    Then from the two distributions estimated fdr from pep
    """
    X = decoy.reshape(-1, 1)
    # label = ["d"] * decoy.shape[0] + ["t"] * target.shape[0]
    # label = np.array(label).reshape(-1, 1)
    clf = GaussianMixture(
        n_components=2,
        covariance_type="full",
        tol=1e-24,
        max_iter=1000,
        random_state=42,
    )
    pred_ = clf.fit(X).predict(X.reshape(-1, 1)).reshape(-1, 1)
    return np.hstack((X, pred_))


def split_posterior(X):
    """
    split classes into tp and fp based on class label after gmm fit
    """
    # force to have tp as max gmm moves label around
    d0 = X[X[:, 1] == 0][:, 0]
    d1 = X[X[:, 1] == 1][:, 0]
    if np.max(d0) > np.max(d1):
        return d0, d1
    else:
        return d1, d0


def fdr_from_pep(tp, fp, target_fdr=0.5):
    """
    estimate fdr from array generated in calc_pdf
    returns estimated fdr at each point of TP and also the go cutoff
    fdr is nr of fp > point / p > point
    """

    def fdr_point(p, fp, tp):
        fps = fp[fp >= p].shape[0]
        tps = tp[tp >= p].shape[0]
        return fps / (fps + tps)

    roll_fdr = np.vectorize(lambda p: fdr_point(p, fp, tp))
    fdr = roll_fdr(fp)
    return fdr, np.percentile(fp, target_fdr * 100)


def estimate_cutoff(fdr_arr, thresh, target_fdr=0.5):
    """
    estimate corum cutoff for target FDR
    use the lowest percentage of corum for reaching target fdr
    """
    fdr2thresh = dict(zip(fdr_arr, thresh))
    fdr_min = min([abs(x - target_fdr) for x in fdr_arr])
    idx = [abs(x - target_fdr) for x in fdr_arr].index(fdr_min)
    return fdr2thresh[fdr_arr[idx]]


def filter_hypo(combined, go_cutoff):
    """
    return object for collapse py
    """
    mask = (combined["ANN"] != 1) & (combined["TOTS"] < go_cutoff)
    filt = combined.drop(combined[mask].index)
    return filt


def eval_complexes(cmplx):
    """
    return appropriate split from database
    use either all positive if more than 50 else use all db
    return None otherwise
    """
    if cmplx[(cmplx["IS_CMPLX"] == "Yes") & (cmplx["ANN"] == 1)].shape[0] > 50:
        # return only positive database
        return cmplx[(cmplx["IS_CMPLX"] == "Yes") & (cmplx["ANN"] == 1)]
        # use all complexes
    elif cmplx[cmplx["ANN"] == 1].shape[0] > 0:
        return cmplx[cmplx["ANN"] == 1]
    else:
        # empty so can quack
        return pd.DataFrame()


def fdr_from_GO(cmplx_comb, target_fdr, fdrfile):
    """
    use positive predicted annotated from db to estimate hypothesis fdr
    """
    pos = cmplx_comb[cmplx_comb["IS_CMPLX"] == "Yes"]
    # remove already here the hypothesis with 0 go
    hypo = pos[(pos["ANN"] != 1) & (pos["TOTS"] > 0)]
    db = cmplx_comb[cmplx_comb["ANN"] == 1]
    db_use = eval_complexes(cmplx_comb)
    io.create_file(fdrfile, ["fdr", "sumGO"])
    if target_fdr > 0:
        thresh = list(pos["TOTS"])
        go_cutoff = 0
        nm = list(hypo.index)
        # if empty then GMM
        if db_use.empty:
            # we update nm here
            print("Not enough reported for FDR estimation, using GMM model")
            # then we need to extract the go sum only
            go_hypo = hypo["TOTS"].values
            predicted = calc_pdf(go_hypo)
            tp, fp = split_posterior(predicted)
            thresh_fdr, go_cutoff = fdr_from_pep(tp=tp, fp=fp, target_fdr=target_fdr)
        else:
            ppi_db = db2ppi(db_use["MB"])
            thresh_fdr, conf_m = calc_fdr(hypo, ppi_db, thresh)
            go_cutoff = estimate_cutoff(thresh_fdr, thresh, target_fdr)
            io.create_file(fdrfile + ".conf_m", ["tp" "fp" "tn" "fn"])
            for pairs in zip(conf_m, thresh):
                io.dump_file(fdrfile + ".conf_m", "\t".join(map(str, pairs)))
        for pairs in zip(thresh_fdr, thresh):
            io.dump_file(fdrfile, "\t".join(map(str, pairs)))
        print("Estimated GO cutoff is {}".format(go_cutoff))
        return filter_hypo(cmplx_comb, go_cutoff), zip(thresh_fdr, thresh, nm)
    else:
        return filter_hypo(cmplx_comb, 0), zip([0], [0], [0])
