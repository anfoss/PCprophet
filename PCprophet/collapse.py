import os
import sys
import re
import operator
from itertools import combinations
import numpy as np
from scipy import interpolate
import pandas as pd

import PCProphet.io_ as io
import PCProphet.stats_ as st
import PCProphet.go_fdr as go_fdr
from PCProphet.exceptions import NotImplementedError


def dedup_cmplx(ref_cmplx, nm):
    """
    get a dict with complex and SORTED protein names inside
    then it check duplicate keys
    this is only for duplicates between corum and hyp generation
    comb is protein_ to id
    """
    comb = io.makehash()
    # sorting is VERY important to keep consistent annotation
    k = list(ref_cmplx.keys())
    k.sort()
    for cm in k:
        prot = ref_cmplx[cm]
        prot.sort()
        nm = "_".join(prot)
        if comb[nm]:
            comb[nm] = comb[nm] + "#" + cm
        else:
            comb[nm] = cm
    # now all duplicates are in comb
    prot2cmplx = {}
    for ids in ref_cmplx:
        prot = ref_cmplx[ids]
        prot.sort()
        prot2cmplx[ids] = comb["_".join(prot)]
    return prot2cmplx, comb


def filter_hypo(ref, hypo, ann):
    """
    receive a hash of hash and remove every keys where annotation is 0 and
    not in hypo
    """
    ann = pd.read_csv(ann, sep="\t")
    #  ann.set_index('ID', inplace=True)
    # corum
    test = dict(zip(ann["ID"], ann["ANN"]))
    hp = [x.replace('"', "") for x in list(hypo.keys())]
    ref2 = {}
    for k in list(ref.keys()):
        k2 = k.replace('"', "")
        if test[k2] == 0 and k2 not in hp:
            pass
        else:
            ref2[k] = ref[k]
    return ref2


def collapse_to_mincal(hypo, peaks, mwuni, cal):
    """
    collapse protein hypothesis to max calibration
    need for getting a protein weight files
    """
    comb_complex = []
    ov = 0.75
    # first sort format the uniprot accession to make it workable
    mw = io.df2dict(mwuni, "Gene names", "Mass")
    mw2 = {}
    for k in mw.keys():
        # mw is in Th
        v = float(mw[k].replace(",", ""))
        if " " in str(k):
            acc = k.split(" ")
            tmp = dict(zip(acc, [v] * len(acc)))
            mw2.update(tmp)
        else:
            mw2[k] = v
    # now we loop through all the possible combination and calc mass
    for k in hypo.values():
        tmp = {x: hypo[x]["V"] for x in hypo.keys() if st.overl(k, hypo[x]) >= ov}
        comb_complex.append(list(tmp.keys()))
    # TODO need to dedup somehow now every row has a dict might become slow
    # peaks has complex => protein => peaks
    # use of complex ID as keys
    out = []
    for cmplx_d in comb_complex:
        # take median of all peaks selected peaks[SEL] in peaks_list.txt
        # for each proposed complex to merge
        # use watermark for performance
        final = 1000000000
        candidate = ""
        for acc in cmplx_d:
            theor = sum([mw2.get(x, 0) for x in peaks[acc].keys()])
            ave_peak = st.medi([int(v.split("\t")[1]) for v in peaks[acc].values()])
            obs = cal[round(ave_peak)]
            if abs(theor - obs) < final:
                candidate = acc
        out.append(candidate)
        # now we end up with list of names
    toret = {}
    for k in out:
        toret[k] = hypo[k]
    return toret


def collapse_to_largest(hypo, ov=0.5):
    """
    collapse to largest subset
    need to calculate overlap members
    """
    # hypo = { 'a':[1,2,3,4], 'b':[2,3], 'd':[7,8,9]}
    m = []
    for k in hypo.values():
        tmp = {x: len(hypo[x]) for x in hypo.keys() if st.overl(k, hypo[x]) >= ov}
        m.append(max(tmp, key=lambda k: tmp[k]))
    # TODO need to change this to return a correctly formatted dict
    toret = io.makehash()
    for ids in hypo:
        toret[ids] = []
        toret[ids].extend(list(hypo[ids]["V"]))
    return toret


def collapse_to_GO_e(hypo, nr=2, subs=lambda x, y: set(x).issubset(y)):
    """
    check GO score for each complex and when decrease stops
    we start from longest keys i.e from the highest branches of dendrogram
    check subsets. smarter to check overlap rather than subsets
    """
    # good keys are in here
    tokeep = io.makehash()
    # print('# hypothesis is ' + str(len(hypo.keys())))
    # now we keep all keys with nr elements and remove them from the hypo
    pairs = {k: v for k, v in hypo.items() if len(v["V"]) == nr}
    shypo = {k: hypo[k] for k in hypo if k not in list(pairs.keys())}
    # print(pairs)
    # assert False
    # for each pairs
    # now let's check if there is any small hypo
    l = len(pairs)
    i = 0
    for k, prots in pairs.items():
        cmplx = {k: v for k, v in shypo.items() if subs(prots["V"], v["V"])}
        if not cmplx:
            # if we are already at the biggest subset let's just add it and keep it there
            i += 1
            tokeep[k] = pairs[k]
        else:
            # query from longest to smallest the highest GO
            sc = max([k["S"] for k in cmplx.values()])
            cmplx = {k: v for k, v in cmplx.items() if v["S"] == sc}
            tm = list(cmplx.keys())[0]
            tokeep[tm] = cmplx[tm]
    # if we did not add anything to the thing let's drop it
    if i == l:
        return hypo
    else:
        print("hypothesis collapsed to " + str(len(tokeep.keys())) + " complexes")
        return collapse_to_GO_e(tokeep, nr + 1)


def collapse(hypo, mode, *args):
    if mode == "GO":
        return reformat_hypo(collapse_to_GO_e(hypo))
    elif mode == "CAL":
        return reformat_hypo(collapse_to_mincal(hypo, *args))
    elif mode == "SUPER":
        return reformat_hypo(collapse_to_largest(hypo))
    else:
        return reformat_hypo(hypo)


def reformat_hypo(hypo):
    toret = io.makehash()
    for ids in hypo:
        toret[ids] = []
        toret[ids].extend(list(hypo[ids]["V"]))
    return toret


def smart_rename(dic):
    """
    assign MW in MDa or KDa depending on the string
    """
    for k in dic:
        if len(dic[k].split(".")[0]) > 6:
            dic[k] = str(round(float(dic[k]) / 1000000, 2)) + " MDa"
        else:
            dic[k] = str(round(float(dic[k]) / 1000, 2)) + " KDa"
    return dic


def calc_calibration(calpath):
    """
    calculate the calibration curve from a file with fraction and
    return a dict fract =>
    """
    fr, mw = io.read_cal(calpath)
    mw = np.array([np.log10(x * 1000) for x in mw])
    fr = np.array(fr)
    f = interpolate.UnivariateSpline(fr, mw, k=1)
    xnew = np.array(list(range(1, 72)))
    cal_d = dict(zip(xnew, f(xnew)))
    cal_d = {k: round(10 ** v, 2) for k, v in cal_d.items()}
    calout = "cal.txt"
    io.create_file(calout, ["FR", "MW"])
    k = smart_rename({str(x): str(v) for x, v in cal_d.items()})
    [io.dump_file(calout, "\t".join([x, k[x]])) for x in list(k.keys())]
    return cal_d


def reformat_annotation(combined_path, simil=0.75):
    """
    uniforms annotation for hypothesis across the samples
    """
    hoa = io.read_combined(combined_path)
    unique = 1
    # filter reported complexes out
    final = {}
    for k in list(hoa.keys()):
        # let's figure out if it is novel or not
        if re.findall(r"^cmplx_+(#cmplx_+)*", k):
            pass
        else:
            # reported
            hoa.pop(k)
            final[k] = k
    m = []
    allk = list(set(hoa.keys()))
    for k in allk:
        m.append([st.overl(hoa[k], hoa[x]) for x in allk])
    m = np.array(m)
    for ind, row in enumerate(m):
        # use index based annotation
        totest = [i for i, x in enumerate(row) if row[i] >= simil]
        # if more than one complex with high overlap
        if len(totest) > 1:
            # if AC & AB simil need to test if BC > simil
            tmp = set()
            tmp.add(allk[ind])
            for pairs in combinations(totest, 2):
                if m[pairs[0]][pairs[1]] >= simil and pairs[0] != pairs[1]:
                    [tmp.add(allk[x]) for x in list(pairs)]
                    m[pairs[0]][pairs[1]] = 0
                    m[pairs[1]][pairs[0]] = 0
                else:
                    # if nothing do not add
                    continue
            # get all unique names
            # create ids
            idx = "cmplx__" + str(unique)
            unique += 1
            # if we already have tmp in final means one complex is shared
            # print(tmp)
            # print(tmp)
            for cmplx in tmp:
                if cmplx in final.keys():
                    # print(cmplx)
                    # print(final)
                    final[cmplx] = final[cmplx] + ";" + idx
                else:
                    final[cmplx] = idx
        elif allk[ind] in final.keys():
            # this key was already added
            pass
        else:
            final[allk[ind]] = allk[ind]
    df = pd.read_csv(combined_path, sep="\t")
    df["CMPLX"] = [final[x] for x in df["CMPLX"]]
    # now we need to split ['CMPLX'] based on ;
    df = (
        df.set_index(df.columns.drop("CMPLX", 1).tolist())
        .CMPLX.str.split(";", expand=True)
        .stack()
        .reset_index()
        .rename(columns={0: "CMPLX"})
        .loc[:, df.columns]
    )
    df.to_csv(combined_path, sep="\t", index=False)


def runner(tmp_, ids, cal, mw, fdr, mode):
    """
    read folder tmp in directory.
    then loop for each file and create a combined file which contains all files
    creates in the tmp directory
    """
    outname = os.path.join(tmp_, "combined.txt")
    header = ["ID", "CMPLX", "COND", "REPL", "PKS", "SEL", "INT", "P", "CMPLT", "GO"]
    io.create_file(outname, header)
    dir_ = []
    dir_ = [x[0] for x in os.walk(tmp_) if x[0] is not tmp_]
    exp_info = io.read_sample_ids(ids)
    strip = lambda x: os.path.splitext(os.path.basename(x))[0]
    exp_info = {strip(k): v for k, v in exp_info.items()}
    wrout = []
    # # calibration info
    try:
        if os.path.isfile(cal):
            cal = calc_calibration(cal)
    except TypeError as e:
        pass
    for smpl in dir_:
        # retrieve all info
        mp_feat_norm = os.path.join(smpl, "mp_feat_norm.txt")
        pred_out = os.path.join(smpl, "rf.txt")
        ann = os.path.join(smpl, "cmplx_combined.txt")
        pred = io.read_pred(pred_out)
        ref_cmplx = io.read_mp_feat(mp_feat_norm)
        prot = io.read_matrix(os.path.join(smpl, "transf_matrix.txt"))
        peak = io.read_peaks(os.path.join(smpl, "peak_list.txt"))
        hyp, _ = go_fdr.fdr_from_GO(
            pred=pred_out,
            db=mp_feat_norm,
            cmplx_ann=ann,
            target_fdr=float(fdr),
            fdrfile=os.path.join(smpl, "fdr.txt"),
        )
        smpl = os.path.basename(os.path.normpath(smpl))
        print(smpl, exp_info[smpl])
        hyp = collapse(hyp, mode, peak, mw, cal)
        ref_cmplx = filter_hypo(ref_cmplx, hyp, ann)
        coll, ref = dedup_cmplx(ref_cmplx, os.path.join(smpl, "collapsed.txt"))
        mapping = io.create_unique(coll)
        coll = io.reformat_dict_f(coll, mapping)
        ref_cmplx = io.reformat_dict_f(ref_cmplx, mapping)
        pred = io.reformat_dict_f(pred, mapping)
        cmplt = io.df2dict(ann, "ID", "CMPLT")
        score = io.df2dict(mp_feat_norm, "ID", "TOTS")
        cmplx_id = list(mapping.keys())
        cmplx_id.sort()
        for ident in cmplx_id:
            cmplx2 = mapping[ident].replace('"', "")
            for mb in ref_cmplx[ident]:
                mb = mb.replace('"', "")
                try:
                    pks = peak[cmplx2][mb]
                    inte = prot[mb]
                    if inte:
                        row = [
                            mb,
                            coll[ident],
                            exp_info[smpl],
                            pks,
                            prot[mb],
                            str(pred[ident]),
                            str(cmplt[cmplx2]),
                            str(score[cmplx2]),
                        ]
                        wrout.append("\t".join(row))
                except KeyError as e:
                    raise e
    [io.dump_file(outname, x) for x in set(wrout)]
    reformat_annotation(outname)
    print("all samples processed")
