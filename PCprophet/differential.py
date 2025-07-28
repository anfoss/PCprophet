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
    dologtrans=True,
    minval=10 ** -17,
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


def extract_inte(df, q=72, norm=False, split_cmplx=False):
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


def differential_(fl, ids):
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
    
    ## now rename and clean output
    return dif_cmplx, dif_prot


def create_complex_report(comb_df, stoic_df, sid_df, outfile):
    def rescale_fr(x, fr):
        try:
            return str(round(x["selected_peak"] * fr[x["condition"]] / 72))
        except ValueError:
            return -1

    # drop single protein now
    comb_df = comb_df[comb_df["rf_probability"] != -1]
    cal = None
    try:
        cal = pd.read_csv("./cal.txt", sep="\t")
        cal = dict(zip([str(round(x)) for x in list(cal["FR"])], cal["MW"]))
    except Exception:
        print("Calibration not provided\nThe MW will not be estimated")
    comb_df = comb_df.drop(["peaks", "rescaled_int", "raw_int", "member"], axis=1)
    # comb_df has exploded all proteins so rows are duplicated extensively.
    # while most things are the same (GO scores, etc) the problem is that the
    # rf confidence and peak selected is different across various replicates.
    # In this way we keep it separated
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
    
    if cal:
        mrg["molecular_weight"] = mrg["selected_peak"]
        mrg.replace({"molecular_weight": cal}, inplace=True)
    else:
        mrg["molecular_weight"] = "0"
    
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
        # cc, mf, bp = set(), set(), set()
        nm = {"CC": set(), "MF": set(), "BP": set()}
        for g in gn.split(":"):
            for onto in gaf[g]:
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
    mrg.to_csv(outfile, index=False)
    return mrg


def create_ppi_report(cmplx_report_out, ppi_report_out):
    """
    create ppi report
    """
    df = pd.read_csv(cmplx_report_out)
    df = df[df['is_complex'] == 'positive']
    #df = df.drop(['rf_probability',"shared_go_cellular_component", "shared_go_biological_process", "shared_go_molecular_function", "selected_peak" , 'completeness', 'go_score', 'molecular_weight', 'stoichiometry', 'in database'], axis=1)
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
    y = df[df["is_complex"] == "Positive"].shape[0]
    if y / df.shape[0] >= thr:
        return "Positive"
    else:
        return "Negative"


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


def runner(infile, sample_ids, outf, temp, dif):
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
    Raises:
        AssertionError: Always raised due to the 'assert False' statement (likely for debugging).
    """
    print(datetime.now())

    if not os.path.isdir(outf):
        os.makedirs(outf)
        
    # stoichiometry calculation    
    comb = pd.read_csv(infile, sep="\t")
    print(comb[comb['complex_id'].str.contains('cmplx')])
    assert False
    # remove single protein
    comb = comb[comb["rf_probability"] != -1]
    stoic_df = stoichiometry(comb, q=3)
    stoic_path = os.path.join(temp, "stoichiometry.txt")
    stoic_df.to_csv(stoic_path, sep="\t", index=False)

    
    sid_df = pd.read_csv(sample_ids, sep="\t")
    comb_df = pd.read_csv(infile, sep="\t")
    cmplx_report_out = os.path.join(outf, "complex_report.csv")
    ppi_report_out = os.path.join(outf, "ppi_report.csv")
    print("Creating complex report and PPI report")
    create_complex_report(comb_df, stoic_df, sid_df, outfile=cmplx_report_out)
    create_ppi_report(cmplx_report_out, ppi_report_out)
    
    ## if there are only controls no differential analysis
    if sid_df["cond"].nunique() == 1 and sid_df["cond"].values[0] == "Ctrl":
        print("No differential analysis performed, only control samples found.")
        return True

    if dif == 'False':    
        ids = dict(zip(sid_df["cond"], sid_df["short_id"]))
        print(datetime.now())

        print("Performing differential analysis for complexes and proteins...")

        dif_cmplx, dif_prot = differential_(infile, ids)
        cmplx_report_df = pd.read_csv(cmplx_report_out)
        cmplx_report_df = cmplx_report_df[
            ["sample_id", "replicate", "is_complex", "complex_id", "members"]
        ]
        # remove single prot accession i.e single ID in the differential complex file
        dif_cmplx = dif_cmplx[~dif_cmplx["ID"].isin(dif_prot["ID"])]
        # this will duplicate the entry
        dif_cmplx = pd.merge(
            cmplx_report_df,
            dif_cmplx,
            left_on=["complex_id", "sample_id"],
            right_on=["ID", "sample_id"],
        ).drop(columns=["ID", "replicate"])
        # count the number of positive complex assignments to assign global
        # assembly state for every condition
        ex = dif_cmplx.groupby(["complex_id"]).apply(assembled).reset_index()
        ex = dict(zip(list(ex["complex_id"]), list(ex[0])))
        dif_cmplx["is_complex"] = dif_cmplx["complex_id"].map(ex)
        dif_cmplx.drop_duplicates(
            subset=["sample_id", "complex_id"], keep="first", inplace=True
        )
        nwnm = {
            "PB4DEX": "probability_differential_assembly_state",
            "LGMLLHN": "assembly_state_log_marginal_likelihood_null",
            "LGMLLHA": "assembly_state_log_marginal_likelihood_alternative",
        }
        dif_cmplx.rename(columns=nwnm, inplace=True)
        dif_cmplx.to_csv(
            os.path.join(outf, "differential_complex_report.csv"), index=False
        )
        dif_prot.rename(columns=nwnm, inplace=True)
        dif_prot.rename(columns={"ID": "gene_name"}, inplace=True)
        dif_prot.to_csv(
            os.path.join(outf, "differential_protein_report.csv"), index=False
        )
        print(datetime.now())
    return True