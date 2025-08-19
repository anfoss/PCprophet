import re
import numpy as np
import pandas as pd
import networkx as nx
import itertools
from collections import namedtuple
from sklearn.mixture import GaussianMixture


def db2ppi(list_sep):
    """
    Read db complexes and convert to PPI graph for faster querying.
    """
    ppi_db = nx.Graph()
    for members in list_sep:
        for pairs in itertools.combinations(members.split("#"), 2):
            ppi_db.add_edge(str.upper(pairs[0]), str.upper(pairs[1]))
    ppi_db.remove_edges_from(nx.selfloop_edges(ppi_db))
    return ppi_db


def overlap_net(ppi_network, mb, over=0.5):
    """
    Calculates overlap between a PPI network and a complex (list of proteins).
    Returns True if >=over fraction of pairs are present in network.
    """
    mb = re.split(r"#", mb)
    if len(mb) < 2:
        return True
    match, nomatch = 0, 0
    for pairs in itertools.combinations(mb, 2):
        if ppi_network.has_edge(pairs[0], pairs[1]):
            match += 1
        else:
            nomatch += 1
    return (match / (nomatch + match)) >= over


def calc_pdf(decoy):
    """
    Fit 2-component Gaussian Mixture to decoy scores.
    Used for fallback FDR estimation when not enough db hits are present.
    """
    X = decoy.reshape(-1, 1)
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
    Split classes into TP-like and FP-like distributions after GMM fit.
    Ensure TP is the higher-scoring distribution.
    """
    d0 = X[X[:, 1] == 0][:, 0]
    d1 = X[X[:, 1] == 1][:, 0]
    if np.max(d0) > np.max(d1):
        return d0, d1
    else:
        return d1, d0


def fdr_from_pep(tp, fp, target_fdr=0.5):
    """
    Estimate FDR from TP and FP distributions.
    FDR(p) = #FP >= p / (#FP >= p + #TP >= p)
    """
    def fdr_point(p, fp, tp):
        fps = fp[fp >= p].shape[0]
        tps = tp[tp >= p].shape[0]
        return fps / (fps + tps) if (fps + tps) else 1.0

    roll_fdr = np.vectorize(lambda p: fdr_point(p, fp, tp))
    fdr = roll_fdr(fp)
    cutoff = np.percentile(fp, target_fdr * 100)
    return fdr, cutoff


def assign_fdr_to_complexes(complexes, db, target_fdr):
    """
    Assign cumulative monotonic FDR to complexes based on GO score.
    
    complexes: pd.DataFrame with columns ['protein_id', score_col, 'members']
               - 'members' must be iterable (list of proteins)
    db: networkX reference network (e.g., CORUM or GO-derived)
    target_fdr: if provided, filter complexes at this cutoff
    """
    # mark if in reference db
    complexes["in_db"] = complexes["members"].apply(lambda m: overlap_net(db, m))

    # sort by score (high → low)
    complexes = complexes.sort_values('TOTS', ascending=False, ignore_index=False)


    # cumulative TP/FP counts
    complexes["tp_cum"] = complexes["in_db"].cumsum()
    complexes["fp_cum"] = (~complexes["in_db"]).cumsum()

    # raw cumulative FDR
    complexes["fdr_raw"] = complexes["fp_cum"] / (complexes["tp_cum"] + complexes["fp_cum"])

    # enforce monotonic non-decreasing FDR
    complexes["fdr"] = np.minimum.accumulate(complexes["fdr_raw"][::-1])[::-1]

    # if filtering at target_fdr
    if target_fdr is not None:
        selected = complexes[complexes["fdr"] <= target_fdr].copy()
    else:
        selected = complexes

    return complexes, selected



def filter_hypo(combined, go_cutoff):
    """
    Filter out hypotheses below the GO cutoff.
    """
    before = combined[combined['reported'] != 1].shape[0]
    mask = (combined["reported"] != 1) & (combined["TOTS"] < go_cutoff)
    filt = combined.drop(combined[mask].index)
    after = filt[filt['reported'] != 1].shape[0]
    print(f"Number of positive complex hypotheses before filtering: {before}")
    print(f"Number of positive complex hypotheses after filtering: {after}")
    return filt


def eval_complexes(cmplx):
    """
    Decide which db complexes to use for FDR estimation.
    Use positives if >50, else all reported.
    """
    if cmplx[(cmplx["is_complex"] == "Yes") & (cmplx["reported"] == 1)].shape[0] > 50:
        return cmplx[(cmplx["is_complex"] == "Yes") & (cmplx["reported"] == 1)]
    elif cmplx[cmplx["reported"] == 1].shape[0] > 0:
        return cmplx[cmplx["reported"] == 1]
    else:
        return pd.DataFrame()


def fdr_from_GO(cmplx_comb, target_fdr, fdrfile):
    """
    Use db-annotated complexes to estimate FDR for hypotheses.
    Returns:
      - filtered complexes (above cutoff)
      - FDR curve dataframe
      - complexes dataframe with per-complex FDR
    """
    pos = cmplx_comb[cmplx_comb["is_complex"] == "Yes"]
    hypo = pos[(pos["reported"] != 1) & (pos["TOTS"] > 0)]
    db_use = eval_complexes(cmplx_comb)

    if target_fdr > 0 and not hypo.empty:
        if db_use.empty or np.all(hypo["TOTS"] == 0):
            print("Not enough reported complexes for FDR estimation, using GMM model")
            go_hypo = hypo["TOTS"].values
            if go_hypo.shape[0] > 0:
                predicted = calc_pdf(go_hypo)
                tp, fp = split_posterior(predicted)
                fdr_values, cutoff = fdr_from_pep(tp=tp, fp=fp, target_fdr=target_fdr)

                # build output dataframe
                df_out = pd.DataFrame({
                    "score": fp,
                    "fdr": fdr_values,
                    "tp": [np.nan] * len(fp),
                    "fp": [np.nan] * len(fp)
                })
                df_out.to_csv(fdrfile, sep="\t", index=False)
                go_cutoff = cutoff
            else:
                print("GO term mapping failed. No FDR control performed.")
                return filter_hypo(cmplx_comb, 0), pd.DataFrame(), cmplx_comb
        else:
            # use cumulative FDR assignment
            ppi_db = db2ppi(db_use["members"])
            complexes_with_fdr, selected = assign_fdr_to_complexes(
                cmplx_comb, ppi_db, target_fdr=target_fdr
            )

            # FDR curve: unique (score, fdr) pairs, descending score
            df_out = complexes_with_fdr[["TOTS", "fdr"]].drop_duplicates().sort_values("TOTS", ascending=False)
            df_out.rename(columns={"TOTS": "score"}, inplace=True)
            df_out.to_csv(fdrfile, sep="\t", index=False)

            go_cutoff = selected["TOTS"].min() if not selected.empty else 0

        # filter complexes by cutoff
        filtered = filter_hypo(cmplx_comb, go_cutoff)

        return filtered, df_out, complexes_with_fdr
    else:
        print("No FDR control performed")
        cmplx_comb["fdr"] = 0
        return filter_hypo(cmplx_comb, 0), pd.DataFrame(), cmplx_comb
