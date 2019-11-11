# !/usr/bin/env python3

import math as m
import sys
import re
import copy
import os
import pandas as pd
import numpy as np

import PCProphet.io_ as io
import PCProphet.stats_ as st
import PCProphet.aligner as aligner
import PCProphet.parse_go as go_parser


def cmplx_stats(cmplx, memb, acc, prot_out, ctrl, treat, score_missing):
    """
    receive single protein complex and compute cmplx statistics
    """
    diff = []
    members = len(list(cmplx))
    mis = 0
    assembly = []
    # now we need to do correlation
    for prot in memb:
        totest_df = cmplx[cmplx["id"] == prot]
        totest_df.drop(["id", "cond"], axis=1, inplace=True)
        if totest_df.shape[0] < 2:
            mis += 1
            diff.append(float(score_missing))
            assembly.append("Missing")
            row = [str(x) for x in [acc, prot, "Missing", 0, 0]]
            io.dump_file(prot_out, "\t".join(row))
        else:
            ccor, cor_n, status = subunit_score(totest_df.iloc[0], totest_df.iloc[1])
            diff.append(cor_n)
            assembly.append(status)
            row = [str(x) for x in [acc, prot, ccor, cor_n, status]]
            io.dump_file(prot_out, "\t".join(row))
    if mis == 0:
        return st.medi(diff), assembly
    else:
        return st.medi(diff) / (members / mis), assembly


def combine_stat(cmplx, mb, cmplx_id, prot_out, ctrl, treat, s):
    """
    receive a complex of 2 conditions with N replicates
    and compute cmplx_stat with combination
    cmplx = object
    acc = name
    prot_out = filename out
    repl = nr of replicates
    """
    # need to filter the proteins and cond in the df
    d = cmplx_stats(
        cmplx=cmplx,
        memb=mb,
        acc=cmplx_id,
        prot_out=prot_out,
        ctrl=ctrl,
        treat=treat,
        score_missing=s,
    )
    comb_desi = d[0]
    comb_status = d[1]

    def common(lst):
        from collections import Counter

        data = Counter(lst)
        return data.most_common(1)[0][0]

    return [comb_desi, common(comb_status)]


def subunit_score(ct, pt):
    """
    calculate cross correlation on centered vectors
    """
    ct_n = (ct - np.mean(ct)) / np.std(ct)
    ct_n = (ct_n - np.min(ct_n)) / (np.max(ct_n) - np.min(ct_n))
    pt_n = (pt - np.mean(pt)) / np.std(pt)
    pt_n = (pt_n - np.min(pt_n)) / (np.max(pt_n) - np.min(pt_n))

    cor_n = np.mean(np.correlate(ct_n, pt_n))
    cor = np.mean(np.correlate(ct, pt))
    status = "Increase"
    if np.mean(pt - ct) < 0:
        status = "Decrease"
    return cor, cor_n, status


def stoichiometry(cmplx, sel):
    """
    receive a protein complex and list of peaks and calculate stoichiometry
    i.e ratio of peak and then rank it lowest to highest
    receive single HoA and sel peaks per protein for condition
    calculate max of sel peaks
    """
    # get values for each peak
    mx = {k: cmplx[k][v] for k, v in sel.items()}
    # now which protein has the max value in the sel peak
    protmax = max(mx, key=mx.get)
    # TODO this can trigger ZeroDivision error
    try:
        ratios = {k: cmplx[k][sel[protmax]] / mx[protmax] for k in sel}
        ratios = {k: ratios[k] for k in ratios if ratios[k] != 0}
        prot, ratio = zip(*ratios.items())
        ratio2 = [round(x / min(ratio), 2) for x in ratio]
        return dict(zip(prot, ratio2))
    except Exception as e:
        return dict(zip(sel.keys(), [1] * len(sel.keys())))


def reformat_cmplx_hoh(cmplx):
    """
    get a complex has HoH and split it
    """
    stoi = []
    for cond in cmplx:
        tmp_stoi = io.makehashlist()
        tmp_prot_nr = []
        for repl in cmplx[cond]:
            pks = {k: cmplx[cond][repl][k]["I"] for k in cmplx[cond][repl]}
            pks2 = io.makehashlist()
            for k in pks:
                pks2[k].extend([float(x) for x in pks[k].split("#")])
            sel = {k: cmplx[cond][repl][k]["C"] for k in cmplx[cond][repl]}
            sel = {k: int(float(v)) for k, v in sel.items()}
            tmp_prot_nr.append(len(sel.keys()))
            dummy = stoichiometry(pks2, sel)
            for pr in dummy.keys():
                tmp_stoi[pr].extend([dummy[pr]])
        mb = round(st.mean(tmp_prot_nr))
        row = "\t".join([cond, average_stoichiometry(tmp_stoi), str(mb)])
        stoi.append(row)
    return stoi


def average_stoichiometry(stoi_dict):
    """
    receive a stoichiometry dict with
    dict[protein] => [stoic, stoic stoich]
    and return the average
    prot:prot:prot = stoic:stoic:stoic
    """
    toret = {}
    for prot in stoi_dict.keys():
        toret[prot] = str(st.mean(stoi_dict[prot]))
    p, s = zip(*toret.items())
    k = [list(x) for x in zip(*sorted(zip(p, s), key=lambda pair: pair[0]))]
    return "\t".join([":".join(k[0]), ":".join(k[1])])


def read_cmplx_data(path, tmp_fold):
    """
    read data in and prepare cmplx array
    """
    header = []
    cmplx, pred_conf = io.makehashset(), io.makedeephash()
    cmplx_stoi = io.makedeephash()
    condition, temp = {}, {}
    prot2int = []
    dummy = []
    for line in open(path, "r"):
        line = line.rstrip("\n")
        if line.startswith(str("ID") + "\t"):
            header = re.split(r"\t+", line)
        else:
            things = re.split(r"\t+", line)
            temp = {}
            temp = dict(zip(header, things))
        # if present this complex let's add
        if temp:
            pr_acc = temp["ID"]
            cond = temp["COND"]
            repl = temp["REPL"]
            # need to be 2d array with cond
            row = [temp["ID"], cond]
            row.extend([float(x) for x in temp["INT"].split("#")])
            prot2int.append(row)
            cmplx[temp["CMPLX"]].add(temp["ID"])
            pred_conf[temp["CMPLX"]][cond][repl] = float(temp["P"])
            cmplx_stoi[temp["CMPLX"]][cond][repl][pr_acc]["I"] = temp["INT"]
            cmplx_stoi[temp["CMPLX"]][cond][repl][pr_acc]["C"] = temp["SEL"]
            condition["_".join([cond, repl])] = cond
            dummy.append(cond)
        else:
            continue
    tmp = []
    for mp in cmplx_stoi:
        tmp.extend([mp + "\t" + x for x in reformat_cmplx_hoh(cmplx_stoi[mp])])
    header = ["CMPLX", "COND", "MB", "RATIO", "NR"]
    stoi_path = os.path.join(tmp_fold, "stoichiometry.txt")
    io.create_file(stoi_path, header)
    [io.dump_file(stoi_path, x) for x in tmp]
    p = average_pred(cmplx, pred_conf, list(set(dummy)))
    return cmplx, p, condition, np.array(prot2int)
    # return average_replicate(cmplx, pred_conf, list(set(condition)))


def pred_score(pred_conf):
    """
    receive a dict with n conditions and prediction and calculate
    the ratio pred c1 / pred c2 (confidence/confidence)
    if pos/pos (0.5 0.5) => p = 1
    min = 1,1 => 0.5 (1/2)
    max = +inf
    use of 1/c1+c2 => 1/(pos/pos) => 1/2 = 0.5 min
    """
    out = io.makehash()
    f = copy.deepcopy(pred_conf)
    # io.create_file('debug.txt', ['nm', 'tmp', 'desi', 'ct', 'tr'])
    for cn in pred_conf:
        # print (pred_conf[cn]['Treat'])
        for cond in pred_conf[cn]:
            if cond == "Ctrl":
                continue
            ctrl = f[cn]["Ctrl"]
            treat = f[cn][cond]
            # both negative
            if ctrl < 0.5 and treat < 0.5:
                out[cn][cond] = 0
            # everything else is good
            else:
                out[cn][cond] = 0.5
    return out


def average_pred(c, pred, cond):
    """
    """
    pred_out = io.makehash()
    for cmplx in c:
        for prot in c[cmplx]:
            # now extract condition
            for condition in cond:
                # let's take care of prediction of complex first
                v = [float(x) for x in list(pred[cmplx][condition].values())]
                pred_out[cmplx][condition] = st.mean(v)
    return pred_score(pred_out)


def calculate_rank(vector):
    """
    rank a list
    """
    a = {}
    rank = 1
    for num in sorted(vector, reverse=True):
        if num not in a:
            a[num] = rank
            rank += 1
    return [a[i] for i in vector]


def test_mb(cmplx, c1, r1):
    """
    receive a cmplx hash and check for each protein the nr of time is detected
    for each condition
    if less than 0.3 then remove
    """
    c1_mb = []
    for prot in cmplx:
        # we are appending for each condition the frequency of the protein
        c1_mb.append((len(list(cmplx[prot][c1]))) / len(r1))
    if len(c1_mb) < 3:
        return 0
    return st.mean(c1_mb)
    # mb_ct = len(list())


def average_profile(a):
    """
    get all replicates for experiment and compute consensus curve
    i.e average fraction wide of all protein
    """
    uniques = np.unique(a, axis=0)
    header = ["id", "cond"]
    header.extend(list(range(0, 72)))
    a_df = pd.DataFrame(uniques, columns=header)
    a_df = a_df.apply(pd.to_numeric, errors="ignore")
    mean_df = a_df.groupby(["id", "cond"]).mean().reset_index()
    return mean_df


def create_complex_report(infile, sto, sid, outfile="ComplexReport.txt"):
    print("Creating complex level report\n")
    sto = pd.read_csv(sto, sep="\t")
    info = pd.read_csv(sid, sep="\t")
    combined = pd.read_csv(infile, sep="\t")
    # test for adding GO
    cal = None
    try:
        cal = pd.read_csv("cal.txt", sep="\t")
        cal = dict(zip([str(round(x)) for x in list(cal["FR"])], cal["MW"]))
    except Exception as e:
        print("Calibration not provided\nThe MW will not be estimated")
    combined.drop(["PKS", "INT", "ID"], inplace=True, axis=1)
    com = combined.groupby(["CMPLX", "COND", "REPL"], as_index=False).mean()
    xx = lambda x, y, on: pd.merge(x, y, how="left", left_on=on, right_on=on)
    on = ["CMPLX", "COND"]
    mrg = xx(sto, com, on)
    # and convert the fraction sel to the new one
    fr = dict(zip(info["cond"], info["fr"]))
    mrg["is complex"] = np.where(mrg["P"] >= 0.5, "Positive", "Negative")
    rescale_fr = lambda x, fr: str(round(x["SEL"] * fr[x["COND"]] / 72))
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
    go = pd.read_csv(io.resource_path("go_terms_class.txt"), sep="\t")
    id2name = dict(zip(go["id"], go["names"]))
    gaf = go_parser.read_gaf_out(io.resource_path("tmp_GO_sp_only.txt"))

    def go_name(gn, gaf, id2name):
        """receive list of gn and split them"""
        # cc, mf, bp = set(), set(), set()
        nm = {"CC": set(), "MF": set(), "BP": set()}
        for g in gn.split(":"):
            for onto in gaf[g]:
                if onto in ["CC", "MF", "BP"]:
                    {nm[onto].add(x) for x in gaf[g][onto].split(";")}
        cc = ";".join([id2name.get(x, x) for x in nm["CC"] if "GO" in x])
        mf = ";".join([id2name.get(x, x) for x in nm["MF"] if "GO" in x])
        bp = ";".join([id2name.get(x, x) for x in nm["BP"] if "GO" in x])
        xx = lambda x: x if x else ""
        return xx(cc), xx(mf), xx(bp)

    cc, mf, bp = [], [], []
    for gn in list(mrg["Members"]):
        try:
            c, m, b = go_name(gn, gaf, id2name)
            cc.append(c)
            mf.append(m)
            bp.append(b)
        except ValueError as e:
            cc.append("")
            mf.append("")
            bp.append("")
    # cc, mf, bp = mrg['Members'].apply(lambda x: )
    mrg["GO Cellular Component"] = cc
    mrg["GO Biological Process"] = bp
    mrg["GO Molecular Function"] = mf
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
            for k in st.fast_comb(np.array(mb), 2):
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


def runner(infile, sample, outf, temp, score_missing, weight_pre, thresh, weights):
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
    """
    if not os.path.isdir(outf):
        os.makedirs(outf)
    ids = io.read_sample_ids_diff(sample)
    aligned_path = os.path.join(temp, "complex_align.txt")
    not_aligned_file = os.path.join(temp, "not_aligned.txt")
    aligner.runner(infile, aligned_path, not_aligned_file)
    cmplx_dic, pred_conf, repl, inte = read_cmplx_data(infile, tmp_fold=temp)
    pred_conf = dict(pred_conf)
    # create report if no differential
    sto = os.path.join(temp, "stoichiometry.txt")
    complex_report_out = os.path.join(outf, "ComplexReport.txt")
    create_complex_report(infile, sto, sample, outfile=complex_report_out)
    ppi_report_out = os.path.join(outf, "PPIReport.txt")
    create_ppi_report(infile=complex_report_out, outfile=ppi_report_out)
    cmplx2prot = io.makehashlist()
    header2 = [
        "ComplexID",
        "GeneName",
        "RawDeltaScore",
        "NormalizedDeltaScore",
        "AssemblyState",
    ]
    header2.extend(["ControlReplicate", "TreatReplicate"])
    ave_prot = average_profile(inte)
    # loop through condition
    for cond, ids in ids.items():
        if cond == "Ctrl":
            continue
        pr_nm = os.path.join(outf, ids + " DifferentialProteinReport.txt")
        io.create_file(pr_nm, header2)
        ds_cmplx, torank, prot = [], [], []
        cm_nm = os.path.join(outf, ids + " DifferentialComplexReport.txt")
        header = ["ComplexID", "PeakScore", "AssemblyState", "Rank", "Members"]
        io.create_file(cm_nm, header)
        print("Calculating complex score for " + str(cond))
        # now we need to get one dataframe with ctrl or cond
        totest_df = ave_prot[ave_prot["cond"].isin(["Ctrl", cond])]
        for acc, cmplx_v in cmplx_dic.items():
            d = 0
            if float(pred_conf.get(acc, 0).get(cond, 0)) > 0:
                d = combine_stat(
                    cmplx=totest_df,
                    mb=cmplx_v,
                    cmplx_id=acc,
                    prot_out=pr_nm,
                    ctrl="Ctrl",
                    treat=cond,
                    s=score_missing,
                )
                pass
            else:
                d = [0, "NA"]
            ds_cmplx.append("\t".join([acc, "\t".join(map(str, d))]))
            torank.append(d[0])
            prot.append(":".join(list(cmplx_v)))
        # now rank condition wide the scores
        rank = calculate_rank(torank)
        fin = [m + "\t" + str(n) + "\t" + k for m, n, k in zip(ds_cmplx, rank, prot)]
        [io.dump_file(cm_nm, x) for x in fin]
