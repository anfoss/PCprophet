import os
import sys
import re
import operator
from itertools import combinations
import numpy as np
from scipy import interpolate
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression


import PCprophet.io_ as io
import PCprophet.stats_ as st
import PCprophet.go_fdr as go_fdr
from PCprophet.exceptions import NotImplementedError


class ProphetExperiment(object):
    """
    docstring for ProphetExperiment
    container for a single PCprophet experiment
    merge all tmp files for a single sec experiment into a complex centric file
    performs fdr calculation
    peaks_c attribute is protein centric
    """
    def __init__(
        self, feature, peaks, pred, prot_matrix, annotation, base, nm, mw=None, cal=None
    ):
        super(ProphetExperiment, self).__init__()
        self.feature = pd.read_csv(feature, sep="\t", index_col="ID")
        self.peaks = pd.read_csv(peaks, sep="\t", index_col="MB", error_bad_lines=False)
        self.pred = pd.read_csv(pred, sep="\t", index_col="ID")
        self.prot_matrix = pd.read_csv(prot_matrix, sep="\t", index_col="ID")
        self.annotation = pd.read_csv(annotation, sep="\t", index_col="ID")
        self.base = base
        self.condition = nm
        self.mw = mw
        self.cal = cal
        self.fdr = None
        self.complex_c = None
        self.peaks_c = None

    def complex_centric_combine(self):
        """
        create combined file using the complexID as index
        """
        self.complex_c = pd.merge(
            self.feature, self.pred, how="inner", left_index=True, right_index=True
        )
        self.complex_c["CREP"] = self.condition
        torm = ["COR", "DIF", "NEG", "SHFT", "W"]
        self.complex_c.drop(torm, inplace=True, axis=1)
        self.complex_c["ANN"] = self.annotation["ANN"]
        self.complex_c["CMPLT"] = self.annotation["CMPLT"]

    def peaks_inte_combine(self):
        """
        combine together peaks and intensity
        prot A peaks detected peaks sel cmplx intensity
        """
        # we need to merge the protein matrix into a single column
        joinall = lambda x: "#".join(x.dropna().astype(str))
        prote = self.prot_matrix.apply(joinall, axis=1)
        self.peaks_c = pd.merge(
            self.peaks, prote.to_frame(), how="inner", left_index=True, right_index=True
        )
        self.peaks_c.rename(columns={0: "INT"}, inplace=True)
        self.peaks_c["CREP"] = self.condition
        return self.peaks_c

    def add_mw(self, mw):
        # force conversion to float
        mw2 = {}
        for k in mw.keys():
            try:
                for k2 in k.split(" "):
                    mw2[k2] = mw[k].replace(",", "")
            except AttributeError as e:
                pass
        self.mw = mw2

    def similarity_graph(self, l, names, ov):
        """
        return a network where every edge between two nodes represents
        jaccard similarity between members > ov
        """

        def min_over(l1, l2):
            inter = len(set(l1).intersection(set(l2)))
            return inter / min(len(l1), len(l2))

        m2 = []
        m = [k.split("#") for k in l]
        for x in m:
            m2.append([min_over(x, y) for y in m])
        arr = np.array(m2)
        possible = np.column_stack(np.where(arr >= ov))
        G = nx.Graph()
        [G.add_edge(names[p[0]], names[p[1]]) for p in possible]
        G.remove_edges_from(G.selfloop_edges())
        return G

    def get_hypo(self):
        """
        returns only positive hypothesis
        """
        pos = self.complex_c[self.complex_c["IS_CMPLX"] == "Yes"]
        return pos[pos["ANN"] != 1]

    def get_db(self):
        """
        returns positive and negative from the database
        """
        return self.complex_c[self.complex_c["ANN"] == 1]

    def get_peaks_inte(self):
        return self.peaks_c

    def interpolate_fract(self):
        """
        calculate fraction to number of compelxes using linear interpolation
        """
        # get db positive
        db_pos = self.get_db()
        db_pos = db_pos[db_pos["IS_CMPLX"] == "Yes"]
        #  calc mean per complex
        # try with highest completness
        db_pos = db_pos[db_pos["CMPLT"] > 0.75]
        peaks2cmplx = self.peaks.groupby("ID").median().round()
        db_pos["sub"] = db_pos["MB"].apply(lambda x: len(x.split("#")))
        cm = pd.merge(peaks2cmplx, db_pos, on=["ID"])
        y, x = cm["sub"].values, cm["SEL"].values
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        peak_dic = dict(zip(list(peaks2cmplx.index), list(peaks2cmplx["SEL"])))
        # actually from here can already get the diff
        theor = {k: p(k) for k in list(range(1, 73))}
        return theor, peaks2cmplx["SEL"]

    def collapse_hypo(self, mode):
        """
        collapse hypothesis using mode
        """
        pos = self.complex_c[self.complex_c["IS_CMPLX"] == "Yes"]
        hypo = pos[pos["ANN"] != 1]
        simil_graph = self.similarity_graph(hypo["MB"], hypo.index, ov=0.5)
        # we need to remove nodes after merging together
        rm = []
        # we need to check the peak position too?
        # better to get db positive here
        lr, peaks = None, None
        if mode == "eCAL":
            lr, peaks = self.interpolate_fract()
        for test in hypo.index.values:
            try:
                tokeep = np.nan
                tomerge = nx.node_connected_component(simil_graph, test)
                simil_graph.remove_nodes_from(tomerge)
                totest = self.complex_c.loc[list(tomerge)]
                if mode == "GO":
                    tokeep = self.collapse_go(totest)
                elif mode == "CAL":
                    tokeep = self.collapse_mincal(totest)
                elif mode == "SUPER":
                    tokeep = self.collapse_largest(totest)
                elif mode == "eCAL":
                    raise NotImplementedError
                rm = np.setdiff1d(np.array(totest.index), np.array(tokeep))
            except KeyError as e:
                # this is always gonna happen because we remove in place
                pass
        # we remove the positive that we are not going to test in differential
        # negatives do not matter
        self.complex_c.drop(index=rm, inplace=True)

    def collapse_largest(self, totest):
        """
        select largest complex
        """
        totest["l"] = totest["MB"].apply(lambda x: len(x.split("#")))
        mx = totest[totest["l"] == totest["l"].max()]
        return mx.index

    def collapse_go(self, totest):
        mx = totest[totest["TOTS"] == totest["TOTS"].max()]
        return mx.index

    def collapse_mincal(self, totest):
        """
        collapse to minimun error from calibration curve
        """
        calc_mw = lambda x, mw: sum([float(mw[gn]) for gn in x.split("#")])
        totest["w"] = totest["MB"].apply(calc_mw, mw=self.mw)
        tmp = self.peaks[self.peaks["ID"].isin(totest.index)]
        tmp = tmp.groupby(["ID"]).mean().SEL.apply(np.round)
        tmp.replace(self.cal, inplace=True)
        diff = (totest["w"] - tmp).abs()
        return diff.idxmin()

    def collapse_empcal(self, totest, lr, sel):
        """
        collapse to empirical calibration using curve fitted with
        average nr subunits and fraction
        need to collect the same nr of fraction and
        """
        raise NotImplementedError
        # TODO need to fix this
        # totest['sub'] = totest['MB'].apply(lambda x: len(x.split('#')))
        # totest['peak'] = sel
        # pk = list(totest['peak'])
        # subs = list(totest['sub'])
        # print(pk)
        # print(lr)
        # [print(lr[x]) for x in pk]
        # totest['d'] = totest.apply(lambda x: abs(lr[x['peaks'] - row['sub']]))
        # print(totest['d'])


    def calc_fdr(self, target_fdr):
        """
        calculate fdr from GO and add FDR to each complex
        """
        fdrfile = os.path.join(self.base, "fdr.txt")
        hyp, fdr = go_fdr.fdr_from_GO(
                                        cmplx_comb=self.complex_c,
                                        target_fdr=float(target_fdr),
                                        fdrfile=fdrfile
                                    )
        fdr = pd.DataFrame(list(fdr), columns=["fdr", "sumGO", "ID"])
        fdr.set_index("ID", inplace=True)
        self.fdr = fdr
        self.complex_c = pd.merge(
            hyp,
            self.fdr.drop(["sumGO"], axis=1),
            how="outer",
            left_index=True,
            right_index=True,
        )
        self.complex_c["fdr"].fillna(0, inplace=True)


class MultiExperiment(object):
    """
    docstring for MultiExperiment
    collapse multiple PCProphetExperiments into a single 'combined.txt'
    file
    """

    def __init__(self):
        super(MultiExperiment, self).__init__()
        self.allexps = []
        self.all_hypo = None
        self.complex_c_all = None
        self.protein_c = None

    def add_exps(self, exp):
        self.allexps.append(exp)

    def multi_collapse(self):
        """
        performs collapsing across multiple ProphetExperiment
        retains only core complexes seen in multiple experiments
        i.e exp 1 A-B-C
        exp 2 A-B-D
        exp3 A-B-C
        keep ABC as most frequent combination of subunits
        """
        annot = 0
        allhypo = pd.concat([exp.get_hypo() for exp in self.allexps])
        # this is only for later splitting to make sure there is no other $
        names = list(allhypo.index + "$" + allhypo["CREP"])
        annot_gr = self.simil_graph_weight(allhypo, names)
        allhypo["nm"] = names
        # now we need to uniform the name across all annotation
        tosub = []
        count = 1
        for test in names:
            try:
                torename = nx.node_connected_component(annot_gr, test)
                annot_gr.remove_nodes_from(torename)
                # select only hypo in torename and rename using cmplx + count
                tmp = allhypo[allhypo["nm"].isin(torename)]
                tmp["ID"] = "cmplx__" + str(count)
                tosub.append(tmp)
            except KeyError as e:
                # remove inplace easier to catch than test has_node
                pass
            finally:
                count += 1
        self.all_hypo = pd.concat(tosub, axis=0)

    def simil_graph_weight(self, hypo, names):
        """
        return a network where every edge between two nodes represents
        weight is overlap  between subunits
        """

        def jaccard(l1, l2):
            s1 = set(l1)
            s2 = set(l2)
            return float(len(s1.intersection(s2))) / float(len(s1.union(s2)))

        #  create a matrix Ncomplex*nfile*nmember
        m2 = []
        m = [k.split("#") for k in hypo["MB"]]
        for x in m:
            m2.append([jaccard(x, y) for y in m])
        arr = np.array(m2)
        possible = np.column_stack(np.where(arr >= 0.5))
        G = nx.Graph()
        [G.add_edge(names[p[0]], names[p[1]]) for p in possible]
        G.remove_edges_from(G.selfloop_edges())
        return G

    def combine_all(self):
        """
        get all hypo and all reported and combine to single file
        """
        alldb = pd.concat([exp.get_db() for exp in self.allexps])
        alldb["nm"] = alldb.index + alldb["CREP"]
        alldb["ID"] = alldb.index
        self.complex_c_all = pd.concat([alldb, self.all_hypo], ignore_index=True)
        return True

    def protein_centric_combine(self):
        """
        explode the rows of the every experiment into all proteins
        """
        self.complex_c_all["MB"] = self.complex_c_all["MB"].str.split("#")
        self.protein_c = io.explode(df=self.complex_c_all, lst_cols=["MB"])
        # nm holds the old name before multi_collapse
        old2new_id = dict(zip(self.protein_c["nm"], self.protein_c["ID"]))
        old2new_id = {k.split("$")[0]: v for k, v in old2new_id.items()}
        self.protein_c.drop(
            ["nm", "IS_CMPLX", "SC_CC", "SC_BP", "SC_MF"], inplace=True, axis=1
        )
        self.protein_c[["COND", "REPL"]] = self.protein_c.CREP.str.split(
            "_", expand=True
        )
        tornm = {"TOTS": "GO", "POS": "P", "ID": "CMPLX", "MB": "ID"}
        self.protein_c.rename(columns=tornm, inplace=True)
        # now add peak and intensity information
        mrg = pd.concat([exp.get_peaks_inte() for exp in self.allexps])
        mrg.replace({"ID": old2new_id}, inplace=True)
        mrg.rename(columns={"ID": "CMPLX"}, inplace=True)
        # need to change the names
        mrg["MB"] = mrg.index
        # is a left merge because mrg also has all complexes not passing fdr
        self.protein_c.drop_duplicates(inplace=True)
        mrg.drop_duplicates(inplace=True)
        self.protein_c = pd.merge(
            self.protein_c,
            mrg,
            how="inner",
            left_on=["CMPLX", "ID", "CREP"],
            right_on=["CMPLX", "MB", "CREP"],
        )
        # reorder to not break differential
        order = [
            "ID",
            "CMPLX",
            "COND",
            "REPL",
            "PKS",
            "SEL",
            "INT",
            "P",
            "CMPLT",
            "GO",
            "CREP",
        ]
        self.protein_c = self.protein_c[order]
        return self.protein_c


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
    return a dict fract
    """
    fr, mw = io.read_cal(calpath)
    mw = np.array([np.log10(x * 1000) for x in mw])
    fr = np.array(fr)
    f = interpolate.UnivariateSpline(fr, mw, k=1)
    xnew = np.array(list(range(1, 73)))
    cal_d = dict(zip(xnew, f(xnew)))
    cal_d = {k: round(10 ** v, 2) for k, v in cal_d.items()}
    calout = "cal.txt"
    io.create_file(calout, ["FR", "MW"])
    k = smart_rename({str(x): str(v) for x, v in cal_d.items()})
    [io.dump_file(calout, "\t".join([x, k[x]])) for x in list(k.keys())]
    return cal_d


def runner(tmp_, ids, cal, mw, fdr, mode):
    """
    read folder tmp in directory.
    then loop for each file and create a combined file which contains all files
    creates in the tmp directory
    """
    dir_ = []
    dir_ = [x[0] for x in os.walk(tmp_) if x[0] is not tmp_]
    exp_info = io.read_sample_ids(ids)
    strip = lambda x: os.path.splitext(os.path.basename(x))[0]
    exp_info = {strip(k): v for k, v in exp_info.items()}
    wrout = []
    try:
        if os.path.isfile(cal):
            cal = calc_calibration(cal)
    except TypeError as e:
        pass
    allexps = MultiExperiment()
    for smpl in dir_:
        base = os.path.basename(os.path.normpath(smpl))
        print(base, exp_info[base])
        mp_feat_norm = os.path.join(smpl, "mp_feat_norm.txt")
        pred_out = os.path.join(smpl, "rf.txt")
        ann = os.path.join(smpl, "cmplx_combined.txt")
        prot = os.path.join(smpl, "transf_matrix.txt")
        peak = os.path.join(smpl, "peak_list.txt")
        exp = ProphetExperiment(
            feature=mp_feat_norm,
            peaks=peak,
            pred=pred_out,
            prot_matrix=prot,
            annotation=ann,
            base=smpl,
            nm=exp_info[base],
            cal=cal,
        )
        if mw is not "None":
            exp.add_mw(io.df2dict(mw, "Gene names", "Mass"))
        exp.complex_centric_combine()
        exp.calc_fdr(fdr)
        exp.collapse_hypo(mode=mode)
        exp.peaks_inte_combine()
        allexps.add_exps(exp)
    allexps.multi_collapse()
    allexps.combine_all()
    final = allexps.protein_centric_combine()
    outname = os.path.join(tmp_, "combined.txt")
    final.to_csv(outname, sep="\t", index=False)
