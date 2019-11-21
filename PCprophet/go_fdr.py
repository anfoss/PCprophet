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
        for pairs in st.fast_comb(members.split('#'), 2):
            ppi_db.add_edge(str.upper(pairs[0]), str.upper(pairs[1]))
    ppi_db.remove_edges_from(ppi_db.selfloop_edges())
    return ppi_db


def get_fdr(tp, fp, tn, fn):
    if fp == 0:
        return 0
    return fp/(tp+fp)


def overlap_net(ppi_network, mb, over=0.5):
    """
    calculates the overlap between a network and a list
    """
    match, nomatch = 0, 0
    mb = re.split(r'#', mb)
    if len(mb) < 2:
        return True
    for pairs in st.fast_comb(mb, 2):
        if ppi_network.has_edge(pairs[0], pairs[1]):
            match +=1
        else:
            nomatch += 1
    if match/(nomatch+match) >= over:
        return True
    else:
        return False


def calc_fdr(combined, db, go_thresh):
    """
    calculate confusion matrix from test and db
    db is a networkX object
    """
    test = dict(zip(list(combined['TOTS']), list(combined['MB'])))
    isindb = {k:overlap_net(db, v) for k, v in test.items()}
    est_fdr = []
    conf_m = []
    for go_score in go_thresh:
        tp, fp, tn, fn = 0,0,0,0
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
        conf_m.append("\t".join(map(str,[tp,fp,tn,fn])))
    return est_fdr, conf_m


def calc_pdf(decoy, target):
    """
    calculate empirical fdr if not enough db hits are present
    use gaussian mixture model 2 components to predict class probability
    for all hypo and db pooled. Then from the two distributions estimated fdr
    from pep

    input == two pandas dataframe with target GO distr and decoy GO distr
    """
    pool = np.concatenate([decoy, target]).reshape(-1,1)
    # remove 0 no need
    X = pool[pool>0].reshape(-1,1)
    # is this needed?? rescale to 1 and then log so higher score gets lower p
    X = -1*np.log(X/np.sum(X))
    label = ['d'] * decoy.shape[0] + ['t'] * target.shape[0]
    label = np.array(label).reshape(-1,1)
    # need to check if converge or not
    clf = GaussianMixture(
                            n_components=2,
                            covariance_type='full',
                            tol = 1e-24,
                            max_iter = 1000
                            # could be reg_covar
                         )
    # NOTE is this needed? at the end we need only the class and the go
    # easy to check classes as np.max(distr1) > np.max(distr2) = distr1 is tp
    logprob = clf.fit(X).score_samples(X)
    posterior = clf.predict_proba(X)
    pdf = np.exp(logprob)
    pdf_individual = posterior * pdf[:, np.newaxis]
    # this predicts class
    pred_ = clf.predict(X.reshape(-1,1)).reshape(-1,1)
    l = np.hstack((posterior[:,1].reshape(-1,1), X, label, pred_))
    # np.savetxt('posterior.csv', l, fmt='%s')
    # now return l splitted in pred tp and pred fp
    return l, posterior

def fdr_from_pep(fp, tp, target_fdr=0.5):
    """
    estimate fdr from array generated in calc_pdf
    returns estimated fdr at each point of TP and also the go cutoff
    fdr is nr of fp > point / p > point
    """
    def fdr_point(p, fp, tp):
        fps = fp[fp>=p].shape[0]
        if fps > 0:
            fps/(fps + tp[tp>p].shape[0])
        else:
            return 0
    fdr = fp.apply_along_axis(lambda p, : fdr_point(p, fp, tp))
    return fdr, np.percentile(fp, target_fdr*100)


def estimate_cutoff(fdr_arr, thresh, target_fdr=0.5):
    """
    estimate corum cutoff for target FDR
    use the lowest percentage of corum for reaching target fdr
    """
    fdr2thresh = dict(zip(fdr_arr, thresh))
    fdr_min = min([abs(x - target_fdr) for x in fdr_arr])
    idx = [abs(x - target_fdr) for x in fdr_arr].index(fdr_min)
    return fdr2thresh[fdr_arr[idx]]


def filter_hypo(hypo, db, go_cutoff):
    """
    return object for collapse py
    """
    mask = ((hypo["ANN"] == 0) & (hypo["TOTS"] >= go_cutoff))
    filt = db.append(hypo[mask], ignore_index=True)
    return filt


def fdr_from_GO(cmplx_comb, target_fdr, fdrfile):
    """
    use positive predicted annotated from db to estimate hypothesis fdr
    """
    pos = cmplx_comb[cmplx_comb['IS_CMPLX']=='Yes']
    hypo = pos[pos['ANN']==0]
    db = pos[pos['ANN']==1]
    ppi_db = db2ppi(db['MB'])

    io.create_file(fdrfile, ['fdr', 'sumGO'])
    io.create_file(fdrfile + '.conf_m', ['tp' 'fp' 'tn' 'fn'])
    # we need to filter for positive
    if hypo.empty or len(set(hypo['TOTS'])) == 1:
        return filter_hypo(hypo, db, 0)
    estimated_fdr = 0.5
    if target_fdr > 0 :
        thresh = list(hypo['TOTS'])
        go_cutoff = 0
        # TODO flow here is quite meh probably better to fail in calc_pdf?
        # if there are more than 100 hypothesis for every reported complex
        if hypo.shape[0]/db.shape[0] > 100 :
            print('Not enough reported for FDR estimation, using GMM model')
            pool, labels = calc_pdf(hypo, decoy)
            thresh_fdr, go_cutoff = fdr_from_pep()
            # if pool:
            #     pass
            #     # now we calculate fdr based on thresh
            # else:
            #     print('GMM fail, a prefixed threshold will be used')
            #     thresh = [0.5]
        else:
            thresh_fdr, conf_m = calc_fdr(hypo, ppi_db, thresh)
            go_cutoff = estimate_cutoff(thresh_fdr, thresh, target_fdr)
            for pairs in zip(conf_m, thresh):
                io.dump_file(fdrfile + '.conf_m', "\t".join(map(str, pairs)))
        nm = list(hypo.index)
        for pairs in zip(thresh_fdr, thresh):
            io.dump_file(fdrfile, "\t".join(map(str, pairs)))
        print("Estimated GO cutoff is {}".format(go_cutoff))
        return filter_hypo(hypo, db, go_cutoff), zip(thresh_fdr, thresh, nm)
    else:
        return filter_hypo(hypo, db, 0), zip([0], [0], [0])
