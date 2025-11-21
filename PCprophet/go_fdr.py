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
    mb = [m.upper() for m in re.split(r"#", mb)]
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
        tol=1e-6,
        max_iter=500,
        random_state=42,
    )
    clf.fit(X)
    labels = clf.predict(X).reshape(-1, 1)
    return np.hstack((X, labels))


def split_posterior(X):
    """
    Split classes into TP-like and FP-like distributions after GMM fit.
    Ensure TP is the higher-scoring distribution.
    """
    d0 = X[X[:, 1] == 0][:, 0]
    d1 = X[X[:, 1] == 1][:, 0]
    if np.mean(d0) > np.mean(d1):
        return d0, d1
    else:
        return d1, d0


def fdr_from_pep(tp, fp, target_fdr=0.5):
    """
    Estimate FDR from TP and FP distributions.
    FDR(p) = #FP >= p / (#FP >= p + #TP >= p)
    """
    if tp.size == 0 and fp.size == 0:
        return np.array([]), 0

    # Evaluate FDR on the union of TP/FP score thresholds (descending)
    thresholds = np.sort(np.unique(np.concatenate([tp, fp])))[::-1]
    tp_counts = np.array([(tp >= t).sum() for t in thresholds], dtype=float)
    fp_counts = np.array([(fp >= t).sum() for t in thresholds], dtype=float)
    fdr_curve = np.where(tp_counts + fp_counts > 0, fp_counts / (tp_counts + fp_counts), 1.0)
    # enforce monotonicity from high score to low score
    fdr_curve = np.minimum.accumulate(fdr_curve[::-1])[::-1]

    threshold_to_fdr = dict(zip(thresholds, fdr_curve))
    fdr = np.array([threshold_to_fdr[x] for x in fp])

    below = thresholds[fdr_curve <= target_fdr]
    cutoff = below[-1] if below.size else thresholds[-1]
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
    print(f"Number of positive hypotheses before filtering: {before}")
    print(f"Number of positive hypotheses after filtering: {after}")
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
    def _with_stub_cols(df, fdr_value=0.0):
        df = df.copy()
        # ensure helper columns and fdr exist
        for col in ["tp_cum", "fp_cum", "fdr_raw"]:
            if col not in df.columns:
                df[col] = 0
        if "fdr" not in df.columns:
            df["fdr"] = fdr_value
        else:
            df["fdr"] = df["fdr"].fillna(fdr_value)
        return df
    
    
    pos = cmplx_comb[cmplx_comb["is_complex"] == "Yes"]
    hypo = pos[(pos["reported"] != 1) & (pos["TOTS"] > 0)]
    db_use = eval_complexes(cmplx_comb)
    if target_fdr > 0 and not hypo.empty:
        if db_use.empty or np.all(hypo["TOTS"] == 0):
            print("Not enough reported complexes for FDR estimation, using GMM model")
            go_hypo = hypo["TOTS"].values
            if go_hypo.shape[0] > 0 and np.unique(go_hypo).shape[0] > 1:
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

            df_out = complexes_with_fdr[["TOTS", "fdr"]].drop_duplicates().sort_values("TOTS", ascending=False)
            df_out.rename(columns={"TOTS": "score"}, inplace=True)
            df_out.to_csv(fdrfile, sep="\t", index=False)

            go_cutoff = selected["TOTS"].min() if not selected.empty else 0

        # filter complexes by cutoff
        complexes_with_fdr = complexes_with_fdr.drop(
            columns=["tp_cum", "fp_cum", "fdr_raw"], errors="ignore"
        )

        filtered = filter_hypo(cmplx_comb, go_cutoff)
        return filtered, df_out, complexes_with_fdr
    else:
        print("No FDR control performed")
        cmplx_comb["fdr"] = 0
        return filter_hypo(cmplx_comb, 0), pd.DataFrame(), cmplx_comb
