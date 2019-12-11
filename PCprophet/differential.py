# !/usr/bin/env python3

import math as m
import sys
import re
import copy
import os
import itertools

import pandas as pd
import numpy as np
import scipy.special as spc
import numpy as np
import pandas as pd
import collections as cl


import PCprophet.io_ as io
import PCprophet.aligner as aligner


import scipy.special as spc
import numpy as np
import pandas as pd
import collections as cl

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
    (C) P. Sykacek 2019 <peter@sykacek.net>
    """

    def yok(y):
        # test whether we have replicates in all levels of y If this
        # is not warranted we can not calculate any statistics.
        lvs = list(set(y))
        ok = len(y) > 1 and len(lvs) > 1
        for cl in lvs:
            ok = ok and np.sum(y == cl) > 1
        return ok

    def __init__(self, modeltype="naive", g=0.8, h=1.5, gam=0.025):
        """
        modeltype: type of combination can be "naive" for a
                conditional independence type combination of
                evidence accross variables in subsets or "full"
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
        a suitable "NULL model" which can be compared agains
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

        # we may now express the loig marginal likelihood:
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
        calculates a dataframe with columns "ID", "LGMLLHN" and
        "LGMLLHA". Every row contains for an id the log marginal
        likelihiood for the NULL model (all samples explained by
        constant mean). and the ALT model (X~p(f(y), sigma)) that
        is we use y as discrete labels to predict X. The expression
        1/(1+exp(LGMLLHN-LGMLLHA)) denotes thus the Bayesian model
        probability for "differential regulation". This probability
        is stored in column "PB4DEX".

        IN
        ids2dat: a dict with ids as keys and DataRec as value.
        m: location used in prior over the (y conditional) means
        PI1: prior probability for ID=1 (i.e. prior for alternative
        more complex model).

        OUT

        lgmrgllhdf: a dataframe with with columns "ID", "LGMLLHN"
                    and "LGMLLHA".  Every row contains for an id
                    the log marginal likelihiood for the NULL model
                    (all samples explained by constant mean). and
                    the ALT model (X~p(f(y), sigma)) that is we use
                    y as discrete labels to predict X. The
                    expression 1/(1+exp(LGMLLHN-LGMLLHA)) denotes
                    thus the Bayesian model probability for
                    "differential regulation". This probability is
                    stored in column "PB4DEX".
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
            if type(self).yok(
                y
            ):  # this call allows overriding yok() by deriving from BayesMANOVA
                lgmrgllha = self.mrgllh(y, X, m=m)
                resdict["LGMLLHA"].append(lgmrgllha)
                # construct the same label for all samples to mimick the
                # "null model".
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
         "differential regulation". This probability is
         stored in column "PB4DEX".
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
                y = pids2dat[pids[0]].y
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
    trg2indmap=True):
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

    (C) P. Sykacek 2019 <peter@sykacek.net>
    """

    allpids = dfrm[pidcol].tolist()
    pids = list(set(allpids))
    allcpx = dfrm[cplxcol].tolist()
    cpx = list(set(allcpx))
    cplx2pids = dict()
    # prepare genrating a dict which maps complex ids to protein ids.
    allpids = np.array(allpids)
    allcpx = np.array(allcpx)
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
    # we finally prepare for every protein ID a DataRec entry in the pids2dat dictionary
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


def score_complexes(dfrm, valcols=list(map("{0}".format, list(range(1, 73))))):
    """
    runs differential bayes manova
    """
    (cplx2pids, pids, Xdfrm, pids2dat) = prepcplxdata(
        dfrm, pidcol="ID", cplxcol="CMPLX", trgcol="COND", valcols=valcols
    )
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

    # bmn is initialised with modeltype="naive". In this case the
    # complex probabilities are calculated from the protein
    # probabilities based on a conditional independence assumption.
    # bpdr_cplx_d = bmn.mrgllh4pcplx(cplx2pids, pids2dat)
    # bpdr_cplx_d = bpdr_cplx_d.loc[~np.isnan(bpdr_cplx_d["PB4DEX"].values), :]
    # bpdr_cplx_d.sort_values("PB4DEX", ascending=False, inplace=True)

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
    return bpdr_prot_zmn, bpdr_cplx_fl


def extract_local_peak(row, q=12):
    """
    extract local peak from SEL column and returns peaks around +-5 fractions
    if not possible extract 10
    """
    # move from fraction to index
    pk = row['SEL']-1
    tmp =  row['INT'].split('#')
    if q > 72/2:
        return tmp
    elif pk < q:
        return tmp[:(q*2)]
    elif row['SEL'] > (72-q):
        return tmp[-(q*2):]
    else:
        return tmp[(pk-q):(pk+q)]


def extract_inte(df, q=72, split_cmplx=False):
    """
    modify combined to extract intensity and returns a df
    """
    if split_cmplx:
        df['CMPLX'] = df['CMPLX'].str.split('#')
        df = io.explode(df=df, lst_cols=['CMPLX'])
    df['pksINT'] = df.apply(lambda x: extract_local_peak(x,q), axis=1)
    vals = list(map("{0}".format, list(range(1, q+1))))
    df[vals] = pd.DataFrame(df.pksINT.values.tolist(), index= df.index)

    #
    df[vals] = df[vals].apply(pd.to_numeric, errors="coerce")
    # fix nan if any
    df[vals] = df[vals].fillna(value=0)
    return df, vals


def create_complex_report(infile, sto, sid, outfile="ComplexReport.txt"):
    def rescale_fr(x, fr):
        try:
            return str(round(x["SEL"] * fr[x["COND"]] / 72))
        except ValueError as e:
            return -1
    print("Creating complex level report\n")
    sto = pd.read_csv(sto, sep="\t")
    info = pd.read_csv(sid, sep="\t")
    combined = pd.read_csv(infile, sep="\t")
    # drop single protein now
    combined = combined[combined['P']!=-1]
    cal = None
    try:
        cal = pd.read_csv("./cal.txt", sep="\t")
        cal = dict(zip([str(round(x)) for x in list(cal["FR"])], cal["MW"]))
    except Exception as e:
        print("Calibration not provided\nThe MW will not be estimated")
    combined.drop(["PKS", "INT", "ID"], inplace=True, axis=1)
    com = combined.groupby(["CMPLX", "COND", "REPL"], as_index=False).mean()
    mrg = pd.merge(sto, com, on=['CMPLX', 'COND'])
    # and convert the fraction sel to the new one
    fr = dict(zip(info["cond"], info["fr"]))
    mrg["is complex"] = np.where(mrg["P"] >= 0.5, "Positive", "Negative")
    mrg["SEL"] = mrg.apply(lambda row: rescale_fr(row, fr), axis=1)
    search = []
    for v in mrg["CMPLX"]:
        if re.findall(r"^cmplx_+|#cmplx_+", v):
            search.append("Novel")
        else:
            search.append("Reported")
    mrg["is in db"] = search
    # and convert the names with infos
    ids = dict(zip(info["cond"], info["short_id"]))
    if cal:
        mrg["MW"] = mrg["SEL"]
        mrg.replace({"MW": cal}, inplace=True)
    else:
        mrg["MW"] = "0"
    mrg.replace({"COND": ids}, inplace=True)
    header = [
        "ComplexID",
        "Condition",
        "Members",
        "Stoichiometry",
        "# Members",
        "Replicate",
        "Apex Peak",
        "Prediction confidence",
        "Completness",
        "GO Score",
        "Is Complex",
        "Reported",
        "Estimated MW",
    ]
    # now rename all the columns
    mrg = mrg.rename(dict(zip(list(mrg), header)), axis=1)
    mrg[['Completness']] = mrg[['Completness']].fillna(value=0)
    mrg.to_csv(outfile, sep="\t", index=False)


def create_ppi_report(infile="ComplexReport.txt", outfile="PPIReport.txt"):
    """
    create ppi report
    """
    header = []
    outf = []
    temp = {}
    w = ["Condition", "Replicate", "Reported"]
    print("Generating network from complexes")
    for line in open(infile, "r"):
        line = line.rstrip("\n")
        if line.startswith(str("ComplexID") + "\t"):
            header = re.split(r"\t+", line)
        else:
            things = re.split(r"\t+", line)
            temp = dict(zip(header, things))
        if temp and temp["Is Complex"] == "Positive":
            mb = temp["Members"].split(":")
            sto = temp["Stoichiometry"].split(":")
            d = dict(zip(mb, sto))
            for k in itertools.combinations(mb, 2):
                tmp = [temp["ComplexID"], k[0], k[1], d[k[0]], d[k[1]]]
                tmp.extend([temp[x] for x in w])
                outf.append("\t".join(tmp))
    header = [
        "ComplexID",
        "ProteinA",
        "ProteinB",
        "StoichiometryA",
        "StoichiometryB",
        "Condition",
        "Replicate",
        "Reported",
    ]
    io.wrout(outf, outfile, header)


def runner(infile, sample, outf, temp):
    """
    args
    0 combined_file,
    1 outfolder,
    2 score for missing proteins,
    3 weight for prediction in combined score

    single protein parameters
    desi thresholds (list)

    weight for single protein scoring
    desi weights()

    group by complex first => complex level
    then by id => protein centric
    then score each condition within that
    return to condition level with intra complex scores and extra complex score
    and then baysean on differential
    """
    if not os.path.isdir(outf):
        os.makedirs(outf)
    ids = io.read_sample_ids_diff(sample)
    aligned_path = os.path.join(temp, "complex_align.txt")
    not_aligned_path = os.path.join(temp, "complex_not_align.txt")
    # aligner.runner(infile, aligned_path, not_aligned=not_aligned_path)
    # cmplx_dic, cmplx_pr, repl = read_cmplx_data(infile, tmp_fold=temp)

    # create report if no differential
    sto = os.path.join(temp, "stoichiometry.txt")
    complex_report_out = os.path.join(outf, "ComplexReport.txt")
    create_complex_report(infile, sto, sample, outfile=complex_report_out)
    ppi_report_out = os.path.join(outf, "PPIReport.txt")
    create_ppi_report(infile=complex_report_out, outfile=ppi_report_out)
    combined, vals = extract_inte(pd.read_csv(infile, sep="\t"), q=72)
    allprot, allcmplx = [], []
    for cnd in ids.keys():
        if cnd != "Ctrl":
            tmp = combined[combined["COND"].isin(["Ctrl", cnd])]
            prot, cmplx = score_complexes(tmp, valcols=vals)
            # use the value i.e short_name in sample_ids.txt
            prot["Condition"] = ids[cnd]
            cmplx["Condition"] = ids[cnd]
            allprot.append(prot)
            allcmplx.append(cmplx)
    allprot = pd.concat(allprot)
    allcmplx = pd.concat(allcmplx)

    # rename columns for readbility
    nwnm = {
        "PB4DEX": "ProbabilityDifferentialRegulation",
        "LGMLLHN": "LogMarginalLikelihood Null",
        "LGMLLHA": "LogMarginalLikelihood Alternative",
    }
    # remove single prot accession i.e single
    allcmplx = allcmplx[~allcmplx['ID'].isin(allprot['ID'])]

    allprot.rename(columns=nwnm, inplace=True)
    allprot.rename(columns={'ID': 'Gene name'}, inplace=True)
    allcmplx.rename(columns=nwnm, inplace=True)
    allprot.rename(columns={'ID': 'Complex identifier'}, inplace=True)
    allprot.to_csv(
        os.path.join(outf, "DifferentialProteinReport.txt"), sep="\t", index=False
    )
    allcmplx.to_csv(
        os.path.join(outf, "DifferentialComplexReport.txt"), sep="\t", index=False
    )
