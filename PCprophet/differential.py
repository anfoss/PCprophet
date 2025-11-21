# !/usr/bin/env python3

import re
import os
import itertools

import pandas as pd
import numpy as np
import scipy.special as spc
import collections as cl
import scipy.stats as sta
from itertools import combinations
import pickle 
from datetime import datetime

import PCprophet.io_ as io
import PCprophet.parse_go as go_parser


# datatype which we use for mapping protein ids to a corresponding
# feature and label representation.
DataRec = cl.namedtuple("DataRec", "X y")


class BayesMANOVA:
    """
    BayesMANOVA provides Bayesian calculations for differential
    regulation for sets of variables. This may be used for identifying
    differentially regulated gene expression time courses or
    differentially regulated proteins based on protein characterising
    feature vectors. The method extends also to biological entities
    (pathways, biological processes or similar). It is in particular
    also useful for assessing differential regulation of protein
    complexes. To avoid excessive model parameters the model
    assumptions are simple (no correlation among the multivariate
    response dimensions). Improvements are possible though time
    consuming.
    """

    def yok(y):
        # test whether we have replicates in all levels of y
        lvs = list(set(y))
        ok = len(y) >= 1 and len(lvs) >= 1
        for cl in lvs:
            ok = ok and np.sum(y == cl) >= 1
        return ok

    def __init__(self, modeltype="full", g=0.8, h=1.5, gam=0.025):
        """
        modeltype: type of combination can be 'naive' for a
                conditional independence type combination of
                evidence accross variables in subsets or 'full'
                for multivariate input vectors.

        g,h:       Gamma prior over noise precision. In multivariate
                settings g and h specify the diagonal noise level.
                Defaults to 0.1, 1

        gam:       A g-prior like multiplicative factor which specifies the
                diagonal precision of the parameter prior.
                Defaults to 1.
        """
        self.modeltype = modeltype
        self.g = g
        self.h = h
        self.gam = gam

    def mrgllh(self, lbls, vals, m=None):
        """
        calculate the marginal log likelihood of a MANOVA type
        model.  under a g-prior like setting. The type of model can
        be adjusted by specifying lbls. If lbls contains only one
        label, there is only one group of samples. This constitites
        a suitable 'NULL model' which can be compared agains
        general groupings of the samples.

        args

        lbls: a discrete vector of groups

        vals: a [nsmpl x ndim] matrix of observations.

        m:  mean in Gaussian prior over Manova parameters

        OUT

        mrgllh: a scalar marginal likelihood.
        """
        grps = list(set(lbls))
        ngroups = len(grps)
        nsampls, nin = vals.shape
        if nsampls != len(lbls):
            raise Exception("Mismatching dimensions!")
        if m is None:
            # we have no mean and use the sample mean as m.
            m = np.mean(vals, axis=0)
        if len(m) != nin:
            raise Exception("Mismatching dimensions!")
        # calculate the xi vectors.
        allxi = []
        alln = []
        for cg in grps:
            # working on group cg
            ntau = sum(lbls == cg)
            alln.append(ntau)
            cxi = (self.gam * m + np.sum(vals[lbls == cg, :], axis=0)) / np.sqrt(
                self.gam + ntau
            )  # correction!!
            allxi.append(cxi.tolist())
        # prepare calculation of the log marginal likelihood
        allxi = np.array(allxi)
        alln = np.array(alln)
        g_ht = self.g + 0.5 * nsampls
        # initialise h_hat with h
        h_ht = np.array([self.h] * nin)
        for d in range(nin):
            # calculate h_ht
            h_ht[d] = h_ht[d] + 0.5 * (
                ngroups * self.gam * m[d] ** 2
                + sum(vals[:, d] ** 2)
                - sum(allxi[:, d] ** 2)
            )

        # we may now express the log marginal likelihood:
        lgmrgllh = nin * (self.g * np.log(self.h) - spc.gammaln(self.g))
        lgmrgllh = lgmrgllh + 0.5 * (
            nin * ngroups * np.log(self.gam) - nin * nsampls * np.log(2 * np.pi)
        )
        lgmrgllh = (
            lgmrgllh
            + np.sum(-0.5 * nin * (self.gam + alln))
            + np.sum(spc.gammaln(g_ht) - g_ht * np.log(h_ht))
        )
        return lgmrgllh

    def mrgllh4ids(self, ids2dat, m=None, PI1=0.5):
        """
        calculates a dataframe with columns 'ID', 'LGMLLHN' and
        'LGMLLHA'. Every row contains for an id the log marginal
        likelihiood for the NULL model (all samples explained by
        constant mean). and the ALT model (X~p(f(y), sigma)) that
        is we use y as discrete labels to predict X. The expression
        1/(1+exp(LGMLLHN-LGMLLHA)) denotes thus the Bayesian model
        probability for 'differential regulation'. This probability
        is stored in column 'PB4DEX'.

        IN
        ids2dat: a dict with ids as keys and DataRec as value.
        m: location used in prior over the (y conditional) means
        PI1: prior probability for ID=1 (i.e. prior for alternative
        more complex model).

        OUT

        lgmrgllhdf: a dataframe with with columns 'ID', 'LGMLLHN'
                    and 'LGMLLHA'.  Every row contains for an id
                    the log marginal likelihiood for the NULL model
                    (all samples explained by constant mean). and
                    the ALT model (X~p(f(y), sigma)) that is we use
                    y as discrete labels to predict X. The
                    expression 1/(1+exp(LGMLLHN-LGMLLHA)) denotes
                    thus the Bayesian model probability for
                    'differential regulation'. This probability is
                    stored in column 'PB4DEX'.
        """
        resdict = {"ID": [], "LGMLLHN": [], "LGMLLHA": [], "PB4DEX": []}
        origm = m
        for cid in ids2dat.keys():
            resdict["ID"].append(cid)
            X = ids2dat[cid].X
            if origm:
                # we are given a mean vector and may have to augment
                # it to fit the no of columns in X
                nrep = int(np.ceil(X.shape[1] / len(origm)))
                m = np.tile(origm, nrep)
            y = ids2dat[cid].y
            if type(self).yok(y):  # this call allows overriding yok
                lgmrgllha = self.mrgllh(y, X, m=m)
                resdict["LGMLLHA"].append(lgmrgllha)
                # construct the same label for all samples to mimick the
                # 'null model'.
                y = np.array([0] * len(y))
                lgmrgllhn = self.mrgllh(y, X, m=m)
                resdict["LGMLLHN"].append(lgmrgllhn)
                # calculate and append the Bayesian indicator probability
                resdict["PB4DEX"].append(
                    1
                    / (
                        1
                        + np.exp(lgmrgllhn + np.log(1 - PI1) - lgmrgllha - np.log(PI1))
                    )
                )
            else:
                resdict["LGMLLHA"].append(None)
                resdict["LGMLLHN"].append(None)
                resdict["PB4DEX"].append(None)
        return pd.DataFrame(resdict)

    def mrgllh4pcplx(self, cplx2pids, pids2dat, m=None, PI1=0.5):
        """
        calculation of marginal log likelihoods of protein
        complexes.  depending on self.modeltype we either calculate
        probabilities for proteins and aggretate them by naive
        bayes or we generate a cplxids2dat dictionary which has in
        X the characteristics of all proteins in the complex column
        stacked to a large X matrix. We may then use mrgllh4ids to
        obtain a dataframe with results directly calculated for
        complexes.

        IN
        cplx2pids: dictionary with complex ids as keys and a list
        of corresponding protein ids as values.
        pids2dat: a dict with protein ids as keys and DataRec as
        value.
        m: location used in prior over the (y conditional) means
        PI1: prior probability for ID=1 (i.e. prior for alternative
        more complex model).

        OUT

        lgmrgllhdf: Df Every row contains for an id
         the log marginal likelihiood for the NULL model
         (all samples explained by constant mean). and
         the ALT model (X~p(f(y), sigma)) that is we use
         y as discrete labels to predict X. The
         expression 1/(1+exp(LGMLLHN-LGMLLHA)) denotes
         thus the Bayesian model probability for
         'differential regulation'. This probability is
         stored in column 'PB4DEX'.
        """
        # we have two modes of operation in dependency of self.modeltype
        if self.modeltype == "naive":
            # naive mode with calculations from conditional
            # independence assumptions between protein expression
            # data from different proteins in the complex.
            resdict = {"ID": [], "LGMLLHN": [], "LGMLLHA": [], "PB4DEX": []}
            # we get the indicator probabilities of protein
            # differential regulation
            pbprtdexdf = self.mrgllh4ids(pids2dat, m=m)
            # in the naive mode these indicator probabilities for
            # protein complexes are combined assuming conditional
            # independence. We loop over all protein complex ids:
            for cpidx in cplx2pids.keys():
                resdict["ID"].append(cpidx)
                pids = cplx2pids[cpidx]
                # collect lgmrgllha from all proteins and add them together
                alllmla = pbprtdexdf.loc[pbprtdexdf["ID"].isin(pids), "LGMLLHA"]
                alllmla = [
                    val for val in alllmla if val is not None and not np.isnan(val)
                ]
                lgmrgllha = np.sum(alllmla)
                resdict["LGMLLHA"].append(lgmrgllha)
                # collect lgmrgllhn from all proteins and add them together
                alllmln = pbprtdexdf.loc[pbprtdexdf["ID"].isin(pids), "LGMLLHN"]
                alllmln = [
                    val for val in alllmln if val is not None and not np.isnan(val)
                ]
                lgmrgllhn = np.sum(alllmln)
                resdict["LGMLLHN"].append(lgmrgllhn)
                # finally express and store the probability in favour
                # of the alternative model.
                resdict["PB4DEX"].append(
                    1
                    / (
                        1
                        + np.exp(lgmrgllhn + np.log(1 - PI1) - lgmrgllha - np.log(PI1))
                    )
                )
            return pd.DataFrame(resdict)
        else:
            # generic mode which combines data rows appropriately to
            # per protein complex features and subsequently assesses
            # differential regulation by calling self.mrgllh4ids
            cpxids2dat = dict()
            for cpidx in cplx2pids.keys():
                # generate the aggregated measurements and labels for cpidx
                pids = cplx2pids[cpidx]
                try:
                    y = pids2dat[pids[0]].y
                except Exception as e:
                    print(pids)
                    print(cpidx)
                    raise e
                X = pids2dat[pids[0]].X
                unqlbs = list(set(y))
                # two dicts for collecting the data.
                Xcol = dict()
                ycol = dict()
                # which we initialise
                for lbl in unqlbs:
                    Xcol[lbl] = X[y == lbl, :]
                    ycol[lbl] = y[y == lbl]
                for pid in pids[1:]:
                    # take current X and y and adjust per label to
                    # the data we collected already for that protein
                    # complex
                    X = pids2dat[pid].X
                    y = pids2dat[pid].y
                    # now we aggregate to unqlbs which remains the
                    # same for all protein ids of the current complex
                    for lbl in unqlbs:
                        cX = X[y == lbl, :]
                        cy = y[y == lbl]
                        ny = min(len(ycol[lbl]), len(cy))
                        Xcol[lbl] = np.column_stack((Xcol[lbl][0:ny, :], cX[0:ny, :]))
                        ycol[lbl] = ycol[lbl][0:ny]
                y = np.concatenate(tuple(ycol.values()))
                X = np.concatenate(tuple(Xcol.values()))
                cpxids2dat[cpidx] = DataRec(X=X, y=y)
            # we have now the data for all protein complexes in place
            # and may use self.mrgllh4ids to obtain the differential
            # expression assessment.
            return self.mrgllh4ids(cpxids2dat, m=m)

# data input
def prepcplxdata(
    dfrm,
    pidcol,
    cplxcol,
    trgcol,
    valcols,
    dologtrans=False,
    # 10 ** -17
    minval=10 ** -13,
    trg2indmap=True,
):
    """
    prepcplxdata prepares proteins or protein complexes to be
    analysed for differential expression.

    IN

    dfrm: input data frame
    pidcol: name of protein id column
    cplxcol: name of protein complex column
    trgcol: target column name.
    valcols: value column name
    dologtrans: boolen flag which controls a simple data
          transformation (we move to log in case of True which
          is the default value).
    minval: enforced minimum value before log transforming the
          data.
    trg2indmap: boolean flag whcih controls whether the target
          labels should be mapped to indices (zero based
          intergers).

    OUT  a tuple
    cplx2pids,: dictionary of protein complexes with protein id lists
    pids,: list of protein ids.
    Xdfrm,: a dataframe with columns pidcol, trgcol and valcols.
        values in valcols are safely transformed to a log scale.
    pids2dat: dictionary which maps pids (protein ids) to DataRec
        objects which describe the data for that protein.
    """
    ## remove single elements from complexes
    dfrm = dfrm.dropna(subset=[cplxcol])
    allpids = dfrm[pidcol].tolist()
    pids = list(set(allpids))
    allcpx = dfrm[cplxcol].tolist()
    cpx = list(set(allcpx))
    cplx2pids = dict()
    # prepare generating a dict which maps complex ids to protein ids.
    allpids = np.array(allpids)
    allcpx = np.array(allcpx)
    # BUG here nans?
    for cplid in cpx:
        cplx2pids[cplid] = allpids[allcpx == cplid].tolist()
    alltrgs = dfrm[trgcol].tolist()
    trgs = list(set(alltrgs))

    Xdfrm = dfrm[[pidcol, trgcol] + valcols].drop_duplicates()
    # map trgcol to integers
    if trg2indmap:
        for ival, trgval in enumerate(trgs):
            Xdfrm.loc[Xdfrm.loc[:, trgcol] == trgval, trgcol] = ival
    if dologtrans:
        # adjust values for MANOVA
        X = np.array(Xdfrm[valcols])
        X[X < minval] = minval
        X = np.log(X)
        Xdfrm[valcols] = X
    # we finally prepare for every protein ID a DataRec entry
    pids2dat = dict()
    for pid in pids:
        X = Xdfrm.loc[Xdfrm[pidcol] == pid, valcols].values
        xmn = np.mean(X, axis=0)
        # we count the number of rows for which the mean of the
        # column is identical to the column value.
        sumid = np.sum(X == xmn, axis=0)
        # to remove constant columns from the data
        X = X[:, sumid != X.shape[0]]
        y = Xdfrm.loc[Xdfrm[pidcol] == pid, trgcol]
        pids2dat[pid] = DataRec(X=X, y=y.values)
    return (cplx2pids, pids, Xdfrm, pids2dat)


def score_complexes(
    dfrm, valcols=list(map("{0}".format, list(range(1, 73)))), mode="protein"
):
    """
    runs differential bayes manova
    """
    (cplx2pids, pids, Xdfrm, pids2dat) = prepcplxdata(
        dfrm, pidcol="member", cplxcol="complex_id", trgcol="condition", valcols=valcols
    )
    if mode == "protein":
        bmn = BayesMANOVA()
        # calculate Bayesian probabilities of differential on protein.
        # default mode with location of the prior mean being the sample location.
        bpdr_prot_d = bmn.mrgllh4ids(pids2dat)
        bpdr_prot_d = bpdr_prot_d.loc[~np.isnan(bpdr_prot_d["PB4DEX"].values), :]
        bpdr_prot_d.sort_values("PB4DEX", ascending=False, inplace=True)
        # use location zero in the prior over Manova coefficients.
        # This is required as m gets internally augmented
        # to match the dimension of the feature vector.
        bpdr_prot_zmn = bmn.mrgllh4ids(pids2dat, m=[0.0])
        bpdr_prot_zmn = bpdr_prot_zmn.loc[~np.isnan(bpdr_prot_zmn["PB4DEX"].values), :]
        bpdr_prot_zmn.sort_values("PB4DEX", ascending=False, inplace=True)
        # bpdr_prot_d is better than assuming zero mean prior
        return bpdr_prot_d
    else:
        # finaly we do a full calculation for complexes. Full calculation
        # generates for every protein coimplex a feature matrix X which
        # contains as columns the features of all proteins which are
        # partr of the complex. The rows in X and y are adjusted auch
        # that we target the minimal number of replicates where data is
        # available for all contributing proteins. The log marginal
        # likelihoods and the Bayesian indicator probabilities of protein
        # complexes which do not have enough data to allow calculations
        # are internally set to numpy.nan.
        bmn = BayesMANOVA(modeltype="full")
        bpdr_cplx_fl = bmn.mrgllh4pcplx(cplx2pids, pids2dat)
        bpdr_cplx_fl = bpdr_cplx_fl.loc[~np.isnan(bpdr_cplx_fl["PB4DEX"].values), :]
        bpdr_cplx_fl.sort_values("PB4DEX", ascending=False, inplace=True)
        return bpdr_cplx_fl


def extract_local_peak(row, q, norm=True):
    """
    extract local peak from selected_peak column and returns peaks around +-q fractions
    if not possible extract 10
    """
    # move from fraction to index
    pk = int(row["selected_peak"])
    tmp = row['rescaled_int'].split("#")
    try :
        tmp = np.array(list(map(float, tmp))).flatten()
    except Exception as e:
        print(e, row)
    if norm:
        tmp = sta.zscore(tmp, ddof=1)
    tmp = list(tmp)
    if q > 72 / 2:
        return tmp
    elif pk < q:
        return tmp[: (q * 2)]
    elif row["selected_peak"] > (72 - q):
        return tmp[-(q * 2) :]
    else:
        return tmp[(pk - q) : (pk + q)]


def extract_inte(df, q=72, norm=True, split_cmplx=False):
    """
    modify combined to extract intensity and returns a df
    """
    if split_cmplx:
        df["complex_id"] = df["complex_id"].str.split("#")
        df = df.explode("complex_id")
    df["pksINT"] = df.apply(lambda x: extract_local_peak(x, q, norm), axis=1)
    vals = list(map("{0}".format, list(range(1, (2 * q) + 1))))
    if q > 72 / 2:
        vals = list(map("{0}".format, list(range(1, q + 1))))
    df[vals] = pd.DataFrame(df.pksINT.values.tolist(), index=df.index)
    df[vals] = df[vals].apply(pd.to_numeric, errors="coerce")
    # fix nan if any
    df[vals] = df[vals].fillna(value=0)
    # remove rows with less than 12 real values
    df.dropna(thresh=12, axis=1, inplace=True)
    return df, vals


def create_complex_report(comb_df, stoic_df, sid_df, outfile):
    def rescale_fr(x, fr):
        try:
            return str(round(x["selected_peak"] * fr[x["condition"]] / 72))
        except ValueError:
            return -1

    # drop single protein now
    comb_df = comb_df[comb_df["rf_probability"] != -1]
    comb_df = comb_df.drop(["peaks", "rescaled_int", "raw_int", "member"], axis=1)
    # comb_df has exploded all proteins so rows are duplicated.
    # while most things are the same (GO scores, etc) the problem is that the
    # rf confidence and peak selected is different across various replicates.
    # TBD if keeping replicates info or not
    com = comb_df.groupby(["complex_id", "condition", "replicate"], as_index=False).head(1)
    mrg = pd.merge(stoic_df, com, on=["complex_id", "condition"])
    mrg["is_complex"] = np.where(mrg["rf_probability"] >= 0.5, "positive", "negative")

    # convert the fraction sel to the new one
    fr = dict(zip(sid_df["cond"], sid_df["fr"]))
    mrg["selected_peak"] = mrg.apply(lambda row: rescale_fr(row, fr), axis=1)
    search = []
    for v in mrg["complex_id"]:
        if re.findall(r"^cmplx_+|#cmplx_+", v):
            search.append("novel")
        else:
            search.append("reported")
    mrg["in_database"] = search
    ids = dict(zip(sid_df["cond"], sid_df["short_id"]))
    
    cal = pd.DataFrame()
    try:
        cal = pd.read_csv("cal_predicted.txt", sep="\t")
    except Exception:
        print("Calibration not provided\nThe MW will not be estimated")

    if len(cal)>0:
        cal['fraction'] = cal['fraction'].astype(int)
        mrg['selected_peak'] = mrg['selected_peak'].astype(int)
        cal = dict(zip(cal['fraction'], cal['molecular_weight_kda']))
        mrg['molecular_weight'] = mrg['selected_peak'].replace(cal)
    else:
        mrg["molecular_weight"] = 0
    
    mrg.rename(columns={"member": "members", 'ratio':'stoichiometry'}, inplace=True)
    mrg["sample_id"] = mrg["condition"].map(ids)
    mrg[["completeness"]] = mrg[["completeness"]].fillna(value=0)
    # add GO terms
    go = pd.read_csv(io.resource_path("meta/go_terms_class.txt"), sep="\t")
    id2name = dict(zip(go["id"], go["names"]))
    with open(io.resource_path('meta/go_gaf.pkl'), "rb") as f:
        gaf = pickle.load(f)


    def go_name(gn, gaf, id2name):
        """
        receive list of GN and converts them back to the gn ontology name
        """
        nm = {"CC": set(), "MF": set(), "BP": set()}
        for g in gn.split(":"):
            for onto in gaf.get(g, ''):
                if onto in ["CC", "MF", "BP"]:
                    {nm[onto].add(x) for x in gaf[g][onto]}
        cc = ";".join([id2name.get(x, x) for x in nm["CC"] if "GO" in x])
        mf = ";".join([id2name.get(x, x) for x in nm["MF"] if "GO" in x])
        bp = ";".join([id2name.get(x, x) for x in nm["BP"] if "GO" in x])
        xx = lambda x: x if x else ""
        return xx(cc), xx(mf), xx(bp)


    cc, mf, bp = [], [], []
    for gn in list(mrg["members"]):
        try:
            c, m, b = go_name(gn, gaf, id2name)
            cc.append(c)
            mf.append(m)
            bp.append(b)
        except ValueError:
            cc.append("")
            mf.append("")
            bp.append("")
    mrg["shared_go_cellular_component"] = cc
    mrg["shared_go_biological_process"] = bp
    mrg["shared_go_molecular_function"] = mf
    
    ### if the complex is reported we 0 the FDR as thiis is not needed
    mrg["fdr"] = np.where(mrg["in_database"] == "reported", 0, mrg["fdr"])
    mrg["is_subcomplex_of"] = mrg.apply(
        lambda row: '' if row["is_subcomplex_of"] == row["complex_id"] else row["is_subcomplex_of"],
        axis=1
    )
    mrg = mrg[mrg['is_complex']=='positive']
    mrg.to_csv(outfile, index=False)
    return mrg


def create_ppi_report(cmplx_report_out, ppi_report_out):
    """
    create ppi report
    """
    df = pd.read_csv(cmplx_report_out)
    df = df[df['is_complex'] == 'positive']
    df = df[['complex_id', 'condition', 'replicate', 'members']]
    df['members'] = df['members'].str.split(':')
    df = df.explode(['members'], ignore_index=True)
    ppi_df = []

    for group_keys, group_df in df.groupby(['complex_id', 'condition', 'replicate']):
        members = list(group_df['members'])        
        for pairs in combinations(members, 2):
            ppi_df.append({
                'complex_id': group_keys[0],
                'condition': group_keys[1],
                'replicate': group_keys[2],
                'proteinA': pairs[0],
                'proteinB': pairs[1],
            })

    ppi_df = pd.DataFrame(ppi_df)
    # deduplicate and aggregate PPIs into complex groups
    ppi_df = ppi_df.groupby(['condition','replicate', 'proteinA', 'proteinB']).agg({
        'complex_id': lambda x: ';'.join(x),
    }).reset_index()
    ppi_df.to_csv(ppi_report_out, index=False)


def assembled(df, thr=0.3):
    """
    test for number of positive complex assignments to assign global assembly state for exp
    """
    return df[df["is_complex"] == "positive"].shape[0] / df.shape[0]


def stoichiometry(df, q=3):
    """
    Calculate stoichiometry of proteins in a complex using AUC-based intensity 
    around selected peak indices.

    For each protein, the intensity profile ("rescaled_int" column) is parsed into an array.
    The selected peak position ("selected_peak" column) is used to extract a window of values
    ± `q` points around the peak. The area under the curve (AUC) in that window 
    is computed by summing the intensities. Stoichiometry is computed as the ratio 
    of each protein's AUC to the maximum AUC in the same condition and replicate group.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with the following columns:
        - 'rescaled_int': string of intensity values separated by '#', length =72.
        - 'selected_peak': selected peak index (1-based or 0-based; assumed 0-based here).
        - 'complex_id': complex identifier.
        - 'condition': experimental condition.
        - 'replicate': replicate ID.
        - 'member'  : protein accession or name.
    
    q : int, default=3
        Number of intensity points to include on each side of the selected peak index
        when computing the AUC. Total window size is `2 * flank + 1`.

    Returns
    -------
    df : pandas.DataFrame
        Original dataframe with two new columns:
        - 'PEAK_AUC': summed intensity around the selected peak ± `q`.
        - 'RATIO': stoichiometric ratio normalized to max AUC within (complex_id, condition, replicate).
    
    df_avg : pandas.DataFrame
        Aggregated dataframe with average stoichiometric ratio for each protein across 
        replicates. Columns: ['complex_id', 'condition', 'member', 'ratio'].

    Notes
    -----
    - Index safety is ensured by clipping the bounds of the intensity array.
    - AUC is computed as a simple sum, assuming intensities are already normalized (e.g., 0-1).

    """
    intensity_array = df['rescaled_int'].str.split('#').apply(lambda x: np.array(x, dtype=float))
    sel_index = df['selected_peak'].astype(float).astype(int)
    
    def extract_auc(arr, peak_idx, q=4):
        left = max(0, peak_idx - q)
        right = min(len(arr), peak_idx + q + 1)
        return np.sum(arr[left:right])

    auc_values = [extract_auc(arr, i) for arr, i in zip(intensity_array, sel_index)]
    df['peak_auc'] = auc_values

    group_cols = ['complex_id', 'condition', 'replicate']
    df['ratio'] = df.groupby(group_cols)['peak_auc'].transform(lambda x: x / x.max())
    stoic_avg = df.groupby(['complex_id', 'condition', 'member'])['ratio'].mean().reset_index()
    stoic_avg['ratio'] = stoic_avg['ratio'].round(1).astype(str)
    stoic_avg = stoic_avg.groupby(['complex_id','condition']).agg({
        'ratio': lambda x: ':'.join(x),
        'member': lambda x: ':'.join(x)
    }).reset_index()
    return stoic_avg


def differential_fc(comb_df, sid_df):
    """
    Computes log2 fold changes (log2FC) of protein abundances between experimental conditions and control, 
    and aggregates these values at both the protein and protein complex levels.
    Args:
        comb_df (pd.DataFrame): DataFrame containing at least the columns 'member', 'complex_id', and 'condition', 
            representing protein complex membership and experimental conditions.
        sid_df (pd.DataFrame): DataFrame where each row corresponds to a sample, with columns 'Sample' (file path to 
            quantification data), 'cond' (condition label), and 'repl' (replicate number).
    Returns:
        diff_prot (pd.DataFrame): DataFrame with columns ['complex_id', 'member', 'condition', 'log2fc_protein'], 
            containing log2 fold changes for each protein member in each complex and condition.
        diff_cmplx (pd.DataFrame): DataFrame with columns ['complex_id', 'condition', 'log2fc_protein'], 
            containing mean log2 fold changes at the complex level for each condition.
    Notes:
        - Assumes input sample files contain columns 'protein_id' and 'gene_name', and are tab-separated.
        - Performs median normalization and log2 transformation on protein abundance values.
        - The control condition is assumed to be labeled 'Ctrl'.
        - Ignores infinite and NaN values during normalization.
    """
    
    def log2_fc(subdf):
        tmp = []
        for x in set(subdf['cond']):
            ctrl = subdf[subdf['cond']=='Ctrl']['value'].mean()
            if x != 'Ctrl':
                treat = subdf[subdf['cond']==x]['value'].mean()
                tmp.append([treat - ctrl, x])
        return pd.DataFrame(tmp)

    def subnan(col, q=0.05, shift=1.8, sd=None):

        """
        select lowest k quantile and create downshifted distribution of 1.8 sigma
        returns series with nan sampled from distr
        """
        np.random.seed(0)
        X = col.values
        qs = X[np.where(X <= np.nanquantile(X, q))]
        mu, sigma = np.mean(qs), np.std(qs)
        if sd == None:
            sd = sigma
        X[np.isnan(X)] = np.random.normal(mu - shift * sigma, sd, size=np.isnan(X).sum())
        return X

    prot_df = []
    for fl in sid_df.itertuples():
        tmp_df = pd.read_csv(fl.Sample, sep='\t')
        tmp_df.set_index(['protein_id', 'gene_name'], inplace=True)
        tmp_df = tmp_df.sum(axis=1).to_frame('value').reset_index()
        tmp_df['cond_repl'] = fl.cond + '$' + str(fl.repl)
        # 4 columns protein_id, gene_name, value, cond_repl
        prot_df.append(tmp_df)
    prot_df = pd.concat(prot_df)
    prot_df = pd.pivot_table(prot_df, values='value', index=['protein_id', 'gene_name'], columns='cond_repl')
    ## median normalized
    ## to test with other ones? Like median polish / quantile?
    prot_df = np.log2(prot_df)
    prot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    md = prot_df.mean(axis=0)
    prot_df = prot_df - md + np.median(md)
    ## need to impute nans maybe? for now this 

    prot_df = pd.melt(prot_df.reset_index(), id_vars=['protein_id', 'gene_name'], var_name='cond_repl', value_name='value')
    prot_df = prot_df.reset_index()
    prot_df[['cond', 'repl']] = prot_df['cond_repl'].str.split('$', expand=True)
    ### this guarantees every protein gets a fold change
    prot_df['value'] = subnan(prot_df['value'])

    ### need to be changed to extract pk
    prot_df = prot_df.groupby(['protein_id', 'gene_name']).apply(log2_fc)
    prot_df.columns = ['log2fc_protein', 'cond']
    prot_df = prot_df.reset_index()
    # only need relationships gn -> complex_id -> cond
    tokeep_df = comb_df[['member', 'complex_id', 'condition']].drop_duplicates()
    ## merge into the combined file
    fc_prot = pd.merge(tokeep_df, prot_df, left_on=['member', 'condition'], right_on=['gene_name', 'cond'], how='left')
    fc_prot = fc_prot[['complex_id', 'member', 'gene_name', 'condition', 'log2fc_protein']]
    fc_prot = fc_prot[fc_prot['condition']!='Ctrl']
    fc_prot.drop(['member'], axis=1, inplace=True)
    # now groupby complex and get complex level fc
    fc_cmplx = fc_prot.groupby(['complex_id', 'condition'])['log2fc_protein'].mean()
    fc_cmplx = fc_cmplx.reset_index()
    fc_cmplx = fc_cmplx[~fc_cmplx['complex_id'].str.startswith('cmplx__')]

    #now need to replace condition with sample ids in fc_cmplx
    # then merge using short_id not condition
    tomp = dict(zip(sid_df['cond'], sid_df['short_id']))
    fc_cmplx['condition'] = fc_cmplx['condition'].map(tomp)
    prot_df['condition'] = prot_df['cond'].map(tomp)
    prot_df.drop(['cond', 'level_2', 'protein_id'], axis=1, inplace=True)
    prot_df = prot_df[prot_df['gene_name'].isin(comb_df['member'])]
    return prot_df, fc_cmplx


def differential_dotp(comb_df, sid_df):
    """
    Computes differential dot product (dotp) similarity scores between control and treatment conditions
    for protein complex members, using a square root angle-based metric.
    The function processes a DataFrame of protein complex quantifications, calculates pairwise dotp scores
    between control and treatment replicates for each protein member, and aggregates these scores at both
    the protein and complex levels. The resulting scores represent the average similarity between the
    abundance profiles of proteins (and complexes) in control versus treatment conditions.
    Parameters
    ----------
    comb_df : pandas.DataFrame
        DataFrame containing quantification data for protein complex members. Expected columns include:
        - 'member': protein or gene identifier
        - 'complex_id': complex identifier
        - 'condition': experimental condition (e.g., 'Ctrl', treatment names)
        - 'replicate': replicate identifier
        - 'rescaled_int': string of intensity values separated by '#'
    sid_df : pandas.DataFrame
        DataFrame mapping condition names to short identifiers. Expected columns:
        - 'cond': original condition name
        - 'short_id': short identifier for the condition
    Returns
    -------
    prot_dotp : pandas.DataFrame
        DataFrame with average dotp scores for each protein member and condition.
        Columns: ['gene_name', 'condition', 'dotp']
    cmplx_dotp : pandas.DataFrame
        DataFrame with average dotp scores for each complex and condition.
        Columns: ['condition', 'complex_id', 'dotp']
    Notes
    -----
    - The dotp score is computed using a square root angle metric, which
      measures the similarity between two abundance profiles, normalized to [0,
      1].
    - Only non-control conditions are included in the complex-level aggregation.
    - Condition names are mapped to short identifiers using `sid_df`.
    """
    def angle_sqrt(s1, s2):
        s1 = np.asarray(s1, dtype=np.float64)
        s2 = np.asarray(s2, dtype=np.float64)

        s1 /= s1.sum()
        s2 /= s2.sum()

        sqrt_s1 = np.sqrt(s1)
        sqrt_s2 = np.sqrt(s2)

        sum_cross = np.dot(sqrt_s1, sqrt_s2)
        sum_left = np.dot(sqrt_s1, sqrt_s1)
        sum_right = np.dot(sqrt_s2, sqrt_s2)

        if sum_left == 0 or sum_right == 0:
            return 0.0

        angle = min(1.0, sum_cross / np.sqrt(sum_left * sum_right))  # clamp for safety
        # this has 1- so 1 max differences
        return 1-(1 - (np.arccos(angle) * 2 / np.pi))

    def process_single_protein(subdf):
        ctrl = subdf[subdf['condition'] == 'Ctrl']
        treat = subdf[subdf['condition'] != 'Ctrl']
        
        # If no control or no treatment data, return empty DataFrame with correct columns
        if ctrl.empty or treat.empty:
           return pd.DataFrame({'dotp': pd.Series(dtype='float64'),
                     'condition': pd.Series(dtype='object')})

        ctrl_dict = {
            rep: np.asarray(vals, dtype=np.float64)
            for rep, vals in zip(ctrl['replicate'], ctrl['rescaled_int'])
        }
        treat_dict = {}
        for cond in treat['condition'].unique():
            treat_df = treat[treat['condition'] == cond]
            treat_dict[cond] = {
                rep: np.asarray(vals, dtype=np.float64)
                for rep, vals in zip(treat_df['replicate'], treat_df['rescaled_int'])
            }

        out = []
        for cond, rep_dict in treat_dict.items():
            for rep_treat, treat_vals in rep_dict.items():
                for rep_ctrl, ctrl_vals in ctrl_dict.items():
                    score = angle_sqrt(treat_vals, ctrl_vals)
                    out.append([score, cond])
        if not out:
            return pd.DataFrame({'dotp': pd.Series(dtype='float64'),
                     'condition': pd.Series(dtype='object')})
  
        df_out = pd.DataFrame(out, columns=['dotp', 'condition'])
        return df_out.groupby('condition', as_index=False)['dotp'].mean()

    ## need to rename dds after
    dd = comb_df.drop_duplicates(['member', 'condition', 'replicate']).copy()
    dd['rescaled_int'] = dd['rescaled_int'].str.split('#')
    dd['rescaled_int'] = dd['rescaled_int'].map(lambda lst: np.fromiter((float(x) for x in lst), dtype=np.float64))
    prot_dotp = dd.groupby('member', group_keys=True).apply(process_single_protein).reset_index()
    dd = comb_df[['member', 'complex_id', 'condition']].drop_duplicates().copy()
    dd =dd[dd['member']!=dd['complex_id']]
    dd = dd[dd['condition']!='Ctrl']
    dd = dd[~dd['complex_id'].str.startswith('cmplx__')]
    ### now need to merge complex in there
    prot_dopt = prot_dotp[['member', 'condition', 'dotp']]
    cmplx_dotp = pd.merge(dd, prot_dotp, how='left', on=['condition', 'member'])
    cmplx_dotp = cmplx_dotp.groupby(['condition', 'complex_id'], as_index=False)['dotp'].mean()
    
    ## now need to replace condition with dict zip whatever
    tomp = dict(zip(sid_df['cond'], sid_df['short_id']))
    cmplx_dotp['condition'] = cmplx_dotp['condition'].map(tomp)
    prot_dotp['condition'] = prot_dotp['condition'].map(tomp)
    
    ## rename
    prot_dotp.rename(columns={"member": "gene_name"}, inplace=True)
    return prot_dotp, cmplx_dotp


def differential_bayes(fl, ids):
    """
    performs differential analysis using first raw profiles (i.e abu)
    and then by using the normalized (asm state)
    this needs to be changed for raw use sum and log2FC while for asm use PS module
    """
    df = pd.read_csv(fl, sep="\t")
    combined, vals = extract_inte(df, norm=False)
    dif_prot, dif_cmplx = [], []
    for cnd in ids.keys():
        if cnd != "Ctrl":
            tmp = combined[combined["condition"].isin(["Ctrl", cnd])]
            prot = score_complexes(tmp, valcols=vals, mode="protein")
            tmp = combined[combined["condition"].isin(["Ctrl", cnd])]
            cmplx = score_complexes(tmp, valcols=vals, mode="cmplx")
            # use the short_name in sample_ids.txt
            prot["sample_id"] = ids[cnd]
            cmplx["sample_id"] = ids[cnd]
            dif_prot.append(prot)
            dif_cmplx.append(cmplx)
    dif_prot = pd.concat(dif_prot)
    dif_cmplx = pd.concat(dif_cmplx)
    nwnm = {
        "PB4DEX": "probability_differential_assembly_state",
        "LGMLLHN": "assembly_state_log_marginal_likelihood_null",
        "LGMLLHA": "assembly_state_log_marginal_likelihood_alternative",
        'sample_id':'condition',
        'log2fc_protein' : 'log2fc_complex'
    }
    dif_cmplx.rename(columns=nwnm, inplace=True)
    dif_prot.rename(columns=nwnm, inplace=True)
    dif_prot.rename(columns={"ID": "gene_name", 'sample_id':'condition'}, inplace=True)

    return dif_cmplx, dif_prot


def runner(infile, sample_ids, outf, temp, dif, mode='complex'):
    """
    Executes the differential analysis workflow for protein complexes and proteins.
    This function performs the following steps:
    1. Creates the output directory if it does not exist.
    2. Calculates stoichiometry from the input file and sample information.
    3. Generates complex and protein-protein interaction (PPI) reports.
    4. If only one sample is present, returns True.
    5. Performs differential analysis for complexes and proteins.
    6. Merges and processes results to remove single protein accessions and duplicate entries.
    7. Checks if complexes are assembled in any condition.
    8. Renames columns for clarity and saves differential reports for complexes and proteins.
    Args:
        infile (str): Path to the input file containing data for analysis.
        sample (str): Path to the sample file or sample identifier.
        outf (str): Output directory where results will be saved.
        temp (str): Temporary directory for intermediate files.
        dif (str): Boolean to do or skip differential analysis
    Returns:
        bool: True if only one sample is present, otherwise None.
    """
    print(datetime.now())

    if not os.path.isdir(outf):
        os.makedirs(outf)
        
    # stoichiometry calculation    
    comb = pd.read_csv(infile, sep="\t", low_memory=False)
    # remove single protein
    comb = comb[comb["rf_probability"] != -1]
    stoic_df = stoichiometry(comb, q=3)
    stoic_path = os.path.join(temp, "stoichiometry.txt")
    stoic_df.to_csv(stoic_path, sep="\t", index=False)

    
    sid_df = pd.read_csv(sample_ids, sep="\t")
    comb_df = pd.read_csv(infile, sep="\t", low_memory=False)
    cmplx_report_out = os.path.join(outf, "complex_report.csv")
    ppi_report_out = os.path.join(outf, "ppi_report.csv")
    print("Creating complex report and PPI report")
    create_complex_report(comb_df, stoic_df, sid_df, outfile=cmplx_report_out)
    if mode == 'complex':
        create_ppi_report(cmplx_report_out, ppi_report_out)
    
    ## if there are only controls no differential analysis
    if sid_df["cond"].nunique() == 1 and sid_df["cond"].values[0] == "Ctrl":
        print("No differential analysis performed, only control samples found.")
        return True

    if dif == 'True':    
        
        cmplx_report_df = pd.read_csv(cmplx_report_out)
        cmplx_report_df = cmplx_report_df[
            ["sample_id", "replicate", "is_complex", "complex_id", "members"]
        ]
        
        ids = dict(zip(sid_df["cond"], sid_df["short_id"]))
        print(datetime.now())

        print("Performing differential analysis for complexes and proteins...")

        ## now need to have one that is for FC 
        ## make sure these have the same format (protein_name, condition, value for prot level and complex_id, member, condition, value for cmplx level)

        fc_prot, fc_cmplx = differential_fc(comb_df, sid_df)
        dotp_prot, dotp_cmplx = differential_dotp(comb_df, sid_df)
        dif_prot = pd.merge(
            fc_prot, 
            dotp_prot, 
            on=['gene_name', 'condition'], 
            how='outer', 
        )
        dif_prot.to_csv(
            os.path.join(outf, "differential_protein_report.csv"), index=False
        )

        dif_cmplx = pd.merge(
            fc_cmplx, 
            dotp_cmplx, 
            on=['complex_id', 'condition'], 
            how='outer', 
        )

        dif_cmplx.to_csv(
            os.path.join(outf, "differential_complex_report.csv"), index=False
        )

        # dif_cmplx, dif_prot = differential_bayes(infile, ids)
        # print(df_cmplx.shape, dif_prot.shape)
        # dif_cmplx.to_csv('test_cmplx.csv')
        # dif_prot.to_csv('test_prot.csv')
        # assert False

        # dif_cmplx = dif_cmplx[~dif_cmplx["ID"].isin(dif_prot["ID"])]
        # # this will duplicate the entry
        # dif_cmplx = pd.merge(
        #     cmplx_report_df,
        #     dif_cmplx,
        #     left_on=["complex_id", "sample_id"],
        #     right_on=["ID", "sample_id"],
        # ).drop(columns=["ID", "replicate"])
        # # count the number of positive complex assignments to assign global
        # # assembly state for every condition
        # ex = dif_cmplx.groupby(["complex_id"]).apply(assembled).reset_index()
        # ex = dict(zip(list(ex["complex_id"]), list(ex[0])))
        # dif_cmplx["percentage_is_complex_replicates"] = dif_cmplx["complex_id"].map(ex)
        # dif_cmplx.drop_duplicates(
        #     subset=["sample_id", "complex_id"], keep="first", inplace=True
        # )

        # dif_cmplx = pd.merge(dif_cmplx, fc_cmplx, on=['complex_id', 'condition'], how='left')
        # dif_prot = pd.merge(dif_prot, fc_prot, on=['gene_name', 'condition'], how='left')
        print(datetime.now())
    return True
