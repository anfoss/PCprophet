import os
from functools import reduce
import numpy as np
from scipy import interpolate
import pandas as pd
import networkx as nx
from datetime import datetime

import PCprophet.io_ as io
import PCprophet.go_fdr as go_fdr
from PCprophet.exceptions import NotImplementedError


class ProphetExperiment(object):
    """
    ProphetExperiment

    A container class for managing a single PCprophet experiment. Handles merging of temporary files, FDR calculation, and collapsing of complexes according to specified arguments.

    Attributes:
        feature (pd.DataFrame): Normalized feature data (mp_feat_norm.txt).
        peaks (pd.DataFrame): List of peaks and selected peak per complex.
        pred (pd.DataFrame): Prediction results from predict.py.
        prot_matrix (pd.DataFrame): Resampled protein matrix (not rescaled).
        raw (pd.DataFrame): Raw rescaled protein matrix.
        annotation (pd.DataFrame): Annotation data.
        base (str): Base path for experiment files.
        condition (str): Experiment condition name.
        mw (dict, optional): Molecular weights from UniProt.
        cal (dict, optional): Calibration data generated from collapse.calc_calibration.
        fdr (pd.DataFrame, optional): False discovery rate results.
        complex_c (pd.DataFrame, optional): Combined complex-centric data.
        peaks_c (pd.DataFrame, optional): Combined peaks and intensity data.

    Methods:
        complex_centric_combine():
            Merge feature and prediction data into a complex-centric file.

        peaks_inte_combine():
            Combine peaks, intensity, and protein matrix data.

        add_mw(mw):
            Add or update molecular weight information.

        similarity_graph(l, names, ov):
            Construct a similarity graph based on Jaccard similarity of complex members.

        get_hypo():
            Retrieve positive hypothesis complexes.

        get_db():
            Retrieve positive and negative complexes from the database.

        get_peaks_inte():
            Get combined peaks and intensity data.

        interpolate_fract():
            Calculate fraction-to-complex number mapping using linear interpolation.

        collapse_hypo(mode):
            Collapse hypothesis complexes using the specified mode.

        collapse_largest(totest):
            Select the largest complex from a group.

        collapse_prob(totest):
            Select the complex with the highest probability.

        collapse_go(totest):
            Select the complex with the highest GO score.

        collapse_mincal(totest):
            Collapse to the complex with minimum calibration error.

        calc_fdr(target_fdr):
            Calculate and assign FDR to each complex.

        add_single_prot(cols):
            Add single protein profiles to the dataset.

        NotImplementedError: If a collapsing method is not implemented.
    """
    def __init__(
        self,
        feature,
        peaks,
        pred,
        prot_matrix,
        raw,
        annotation,
        base,
        nm,
        mw=None,
        cal=None,
    ):
        super(ProphetExperiment, self).__init__()
        self.feature = pd.read_csv(feature, sep="\t")
        self.peaks = pd.read_csv(peaks, sep="\t", index_col="member")
        self.pred = pd.read_csv(pred, sep="\t")
        # this is important because peaks_inte_combine uses the index and
        # self.peaks is with gene names ('member') as index
        self.prot_matrix = pd.read_csv(prot_matrix, sep="\t", index_col="gene_name")
        self.raw = pd.read_csv(raw, sep="\t", index_col="gene_name")
        self.annotation = pd.read_csv(annotation, sep="\t", index_col="protein_id")
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
        complex_c = pd.merge(
            self.feature[["complex_id", "members", "SC_CC", "SC_MF", "SC_BP", "TOTS"]], self.pred, how="inner",on='complex_id')
        complex_c["cond_rep"] = self.condition
        # old ANN
        complex_c["reported"] = [0 if x.startswith('hypo') else 1 for x in complex_c['complex_id']]
        
        # old CMPLT
        cmplt = dict(zip(self.annotation['complex_id'], self.annotation['completeness'].fillna(-1)))
        complex_c["completeness"] = complex_c["complex_id"].map(cmplt)
        complex_c.set_index('complex_id', inplace=True)
        self.complex_c = complex_c

    def peaks_inte_combine(self):
        """
        Combine peaks, intensity (prot_matrix), and raw signal into a single DataFrame.
        Output DataFrame has columns: peaks + rescaled_int + raw_int + cond_rep.
        """

        joinall = lambda x: "#".join(x.dropna().astype(str))

        prote = self.prot_matrix.drop(columns=["protein_id"], errors='ignore')
        raws = self.raw.drop(columns=["protein_id"], errors='ignore')

        # Collapse prot_matrix and raw by row
        prote = prote.apply(joinall, axis=1).rename("rescaled_int")
        raws = raws.apply(joinall, axis=1).rename("raw_int")

        # Ensure consistent index types to avoid merge failures
        for df in [self.peaks, prote, raws]:
            df.index = df.index.astype(str)

        # Merge everything on index
        peaks_c = self.peaks.copy()
        peaks_c = peaks_c.merge(prote, left_index=True, right_index=True, how="inner")
        peaks_c = peaks_c.merge(raws, left_index=True, right_index=True, how="inner")

        # Add condition/replicate info
        peaks_c["cond_rep"] = self.condition
        self.peaks_c = peaks_c
        return peaks_c

    def add_mw(self, mw):
        self.mw = mw

    def similarity_graph(self, l, names, ov=0.5):
        """
        Return a network where every edge represents
        pairwise similarity > ov (based on Jaccard over min size).
        
        l: list of '#' concatenated member strings
        names: list of node names (same length as l)
        ov: minimum overlap ratio (e.g. 0.5)
        """

        def min_over(l1, l2):
            inter = len(set(l1).intersection(set(l2)))
            return inter / min(len(l1), len(l2))

        m = [k.split("#") for k in l]
        G = nx.Graph()

        for i in range(len(m)):
            G.add_node(names[i])
            for j in range(i + 1, len(m)):
                if min_over(m[i], m[j]) >= ov:
                    G.add_edge(names[i], names[j])
        return G


    def get_hypo(self):
        """
        returns only positive hypothesis
        """
        pos = self.complex_c[self.complex_c["is_complex"] == "Yes"]
        return pos[pos["reported"] != 1]

    def get_db(self):
        """
        returns positive and negative from the database
        """
        return self.complex_c[self.complex_c["reported"] == 1]

    def get_peaks_inte(self):
        return self.peaks_c

    def interpolate_fract(self):
        """
        calculate fraction to number of compelxes using linear interpolation
        """
        # get db positive
        db_pos = self.get_db()
        db_pos = db_pos[db_pos["is_complex"] == "Yes"]
        #  calc mean per complex
        # try with highest completness
        db_pos = db_pos[db_pos["completeness"] > 0.75]
        peaks2cmplx = self.peaks.groupby("protein_id").median().round()
        db_pos["sub"] = db_pos["members"].apply(lambda x: len(x.split("#")))
        cm = pd.merge(peaks2cmplx, db_pos, on=["protein_id"])
        y, x = cm["sub"].values, cm["selected_peak"].values
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        # peak_dic = dict(zip(list(peaks2cmplx.index), list(peaks2cmplx["selected_peak"])))
        theor = {k: p(k) for k in list(range(1, 73))}
        return theor, peaks2cmplx["selected_peak"]


    ## need to be refactored
    def collapse_hypo(self, mode):
        """
        collapse hypothesis using mode
        """
        self.complex_c.dropna(subset=["members"], inplace=True)
        pos = self.complex_c[self.complex_c["is_complex"] == "Yes"]
        hypo = pos[pos["reported"] != 1]
        simil_graph = self.similarity_graph(hypo["members"], hypo.index)
        # we need to remove nodes after merging together
        print("Collapsing complexes using mode: {}".format(mode))
        rm = []
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
                elif mode == "PROB":
                    tokeep = self.collapse_prob(totest)
                elif mode == "eCAL":
                    raise NotImplementedError
                elif mode == "NONE":
                    tokeep = totest.index
                # idxs to remove
                tm = np.setdiff1d(np.array(totest.index), np.array(tokeep))
                rm.extend(list(tm))
            except KeyError as e:
                # this is always gonna happen because we remove in place
                pass
        self.complex_c.drop(index=rm, inplace=True)
        # print("Removed {} overlapping complexes".format(len(rm)))
        print("Number of complexes after collapsing: {}".format(self.complex_c.index.nunique()))

    def collapse_largest(self, totest):
        """
        select largest complex
        """
        totest["l"] = totest["members"].apply(lambda x: len(x.split("#")))
        mx = totest[totest["l"] == totest["l"].max()]
        return mx.index

    def collapse_prob(self, totest):
        """
        select complex with higest probability per dendrogram branch
        """
        mx = totest[totest["rf_probability"] == totest["rf_probability"].max()]
        return mx.index

    def collapse_go(self, totest):
        mx = totest[totest["TOTS"] == totest["TOTS"].max()]
        return mx.index

    def collapse_mincal(self, totest):
        """
        collapse to minimun error from calibration curve
        """
        ### if not available the mw is estimated at 50 Kda for every protein missing
        calc_mw = lambda x, mw: sum([mw.get(gn, 50000) for gn in x.split("#")])
        totest["w"] = totest["members"].apply(calc_mw, mw=self.mw)
        print(totest.head(10))
        assert False
        tmp = self.peaks[self.peaks["protein_id"].isin(totest.index)]
        tmp = tmp.groupby(["protein_id"]).mean().selected_peak.apply(np.round)
        tmp.replace(self.cal, inplace=True)
        diff = (totest["w"] - tmp).abs()
        return diff.idxmin()
    
    def collapse_ecal(self, totest):
        pass
    
    
    def calc_fdr(self, target_fdr):
        """
        calculate fdr from GO and add FDR to each complex
        """
        fdrfile = os.path.join(self.base, "fdr.txt")
        hyp, fdr_curve, cmplx_with_fdr = go_fdr.fdr_from_GO(
            cmplx_comb=self.complex_c, 
            target_fdr=float(target_fdr), 
            fdrfile=fdrfile
        )
        self.fdr = fdr_curve
        self.complex_c = cmplx_with_fdr.drop(['tp_cum', 'fp_cum', 'fdr_raw'], axis=1)        
        self.complex_c.fillna({'fdr': 0}, inplace=True)

    def add_single_prot(self, cols):
        """
        add single proteins profile to the file
        check duplicate complexes and create unique identifier
        add protein trace with complexID == protname so match between condition
        for removal use rf_probability != -1
        we add the index of max arr as peak
        """
        df = pd.DataFrame(columns=cols, index=self.raw.index)
        df["member"] = self.raw.index
        mrg = lambda x: reduce(lambda a, b: str(a) + "#" + str(b), x)
        raw = self.raw.copy(deep=True)
        raw.drop('protein_id', axis=1, inplace=True)
        df["raw_int"] = raw.apply(mrg, axis=1)
        df["complex_id"] = df["member"].copy(deep=True)
        df["rf_probability"] = -1
        joinall = lambda x: "#".join(x.dropna().astype(str))
        prote = self.prot_matrix.drop(['protein_id'], axis=1).apply(joinall, axis=1)
        df["rescaled_int"] = prote
        df["cond_rep"] = self.condition
        df["is_subcomplex_of"] = np.nan
        df[["condition", "replicate"]] = df.cond_rep.str.split("_", expand=True)
        df["selected_peak"] = raw.apply(lambda x: np.argmax(x), axis=1)
        df[["peaks", "completeness", "go_score"]] = 0
        return df.reset_index(drop=True)



class MultiExperiment(object):
    """
    docstring for MultiExperiment
    collapse multiple PCProphetExperiments into a single 'combined.txt'
    """

    def __init__(self):
        super(MultiExperiment, self).__init__()
        self.allexps = []
        self.all_hypo = None
        self.complex_c_all = None
        self.protein_c = None
        self.common_hypo = {}

    def add_exps(self, exp):
        self.allexps.append(exp)

    def multi_collapse(self):
        """
        Collapses and consolidates core complexes across multiple ProphetExperiment instances.
        This method identifies and retains only the most frequent core complexes (i.e., combinations of subunits)
        that are observed across multiple experiments. For example, if three experiments yield the following complexes:
            - Experiment 1: A-B-C
            - Experiment 2: A-B-D
            - Experiment 3: A-B-C
        The method will retain A-B-C as it is the most frequently observed combination.
        The process involves:
            - Concatenating hypothetical complexes from all experiments.
            - Assigning unique names to each complex for graph-based analysis.
            - Building a similarity graph to group related complexes.
            - Renaming and consolidating complexes that are connected in the graph.
            - Storing the consolidated complexes in `self.all_hypo`.
        If no complexes are found, `self.allhypo` is set to an empty DataFrame.
        Returns:
            None
        """
        # TODO needs to save somewhere a dict of complex_ids to new complex ids name
        allhypo = pd.concat([exp.get_hypo() for exp in self.allexps])
        # this is only for later splitting to make sure there is no other $
        names = [f"{idx}${cond_rep}" for idx, cond_rep in zip(allhypo.index, allhypo["cond_rep"])]
        allhypo["nm"] = names
        annot_gr = self.simil_graph_weight(allhypo, names)
        # now we need to uniform the name across all annotation
        tosub = []
        count = 1
        for test in names:
            try:
                torename = nx.node_connected_component(annot_gr, test)
                annot_gr.remove_nodes_from(torename)
                # select only hypo in torename and rename using cmplx + count
                tmp = allhypo[allhypo["nm"].isin(torename)].copy()
                tmp["complex_id"] = "cmplx__" + str(count)
                
                ## fix the annotation by having a shared dict old name -> new name for self.protein_c
                nm = [x.split('$')[0] for x in torename]
                nm = dict(zip(nm, ["cmplx__" + str(count)]*len(nm)))
                self.common_hypo.update(nm)
                tosub.append(tmp)
            except KeyError:
                # remove inplace faster to catch than test has_node
                pass
            finally:
                count += 1
        if tosub:
            self.all_hypo = pd.concat(tosub, axis=0)
        else:
            self.all_hypo = pd.DataFrame()

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
        m = [k.split("#") for k in hypo["members"]]
        for x in m:
            m2.append([jaccard(x, y) for y in m])
        arr = np.array(m2)
        possible = np.column_stack(np.where(arr >= 0.5))
        G = nx.Graph()
        [G.add_edge(names[p[0]], names[p[1]]) for p in possible]
        G.remove_edges_from(nx.selfloop_edges(G, keys=True))
        return G

    def combine_all(self):
        """
        get all hypo and all reported and combine to single file
        """
        alldb = pd.concat([exp.get_db() for exp in self.allexps])
        alldb["nm"] = alldb.index + alldb["cond_rep"]
        alldb["complex_id"] = alldb.index        
        self.complex_c_all = pd.concat([alldb, self.all_hypo], ignore_index=True)
        

    def protein_centric_combine(self):
        """
        explode the rows of the every experiment into all proteins
        """
        self.complex_c_all["members"] = self.complex_c_all["members"].str.split("#")
        self.protein_c = self.complex_c_all.explode("members")
        # nm holds the old cmplx name before multi_collapse
        old2new_id = dict(zip(self.protein_c["nm"], self.protein_c["complex_id"]))
        old2new_id = {k.split("$")[0]: v for k, v in old2new_id.items()}
        self.protein_c.drop(
            ["nm", "is_complex", "SC_CC", "SC_BP", "SC_MF"], inplace=True, axis=1
        )
        self.protein_c[["condition", "replicate"]] = self.protein_c.cond_rep.str.split(
            "_", expand=True
        )
        self.protein_c.rename(columns= {"TOTS": "go_score", 'members': 'member'}, inplace=True)
        # now add peak and intensity information
        mrg = pd.concat([exp.get_peaks_inte() for exp in self.allexps])
        mrg.replace({"protein_id": old2new_id}, inplace=True)
        mrg.rename(columns={"protein_id": "complex_id"}, inplace=True)
        mrg.reset_index(inplace=True)
        mrg['complex_id'] = mrg['complex_id'].replace(self.common_hypo)
        # is a left merge because mrg also has all complexes not passing fdr
        self.protein_c.drop_duplicates(inplace=True)
        mrg.drop_duplicates(inplace=True)
        self.protein_c = pd.merge(
            self.protein_c,
            mrg,
            how="inner",
            on=["complex_id", "member", "cond_rep"],
        )
        # reorder to not break differential\
        order = [
            "member",
            "complex_id",
            "condition",
            "replicate",
            "peaks",
            "selected_peak",
            "rescaled_int",
            "rf_probability",
            "completeness",
            "go_score",
            "cond_rep",
            "raw_int",
            'is_subcomplex_of'
        ]
        # # now add all single protein accession from each matrix if not present
        # self.protein_c = self.protein_c[order]
        # allprot = pd.concat([x.add_single_prot(order) for x in self.allexps])
        allprot = pd.concat([x.add_single_prot(order) for x in self.allexps])

        allprot = allprot[~allprot["member"].isin(self.protein_c["member"])]
        self.protein_c = pd.concat([self.protein_c, allprot], ignore_index=True)
        return self.protein_c
    
    def calc_subcomplexes(self):
        self.complex_c_all["member_set"] = self.complex_c_all["members"].str.split("#").map(set)
        ids = self.complex_c_all["complex_id"].values
        sets = self.complex_c_all["member_set"].values
        subcomplex_targets = []

        for i, (id_i, set_i) in enumerate(zip(ids, sets)):
            candidates = []
            for j, (id_j, set_j) in enumerate(zip(ids, sets)):
                if i == j:
                    continue
                if set_i.issubset(set_j):
                    candidates.append((len(set_j), id_j))
            if candidates:
                # Select the smallest superset
                subcomplex_targets.append(sorted(candidates)[0][1])
            else:
                subcomplex_targets.append(None)
        self.complex_c_all["is_subcomplex_of"] = subcomplex_targets
        self.complex_c_all.drop(columns=["member_set"], inplace=True)
        

def calc_calibration(calpath):
    """
    calculate the calibration curve from a file with fraction and
    return a dict fract
    """
    import matplotlib.pyplot as plt
    
    from sklearn.linear_model import LinearRegression
    calp = pd.read_csv(calpath, sep='\t')
    fr, mw = list(calp['fraction_number']), list(calp['molecular_weight_kda'])
    mw = np.array([np.log10(x*1000) for x in mw]).reshape(-1, 1)
    fr = np.array(fr).reshape(-1, 1)
    lr = LinearRegression().fit(fr.reshape(-1,1), mw.reshape(-1,1))
    xnew = list(range(1, 73))
    print('R2 score for calibration regression is {}'.format(lr.score(fr ,mw)))
    coef = lr.coef_[0][0]
    inter = lr.intercept_[0]
    # Kda
    calcfr = lambda x: (10**(coef*x + inter)) / 1000
    cal_d = pd.DataFrame({'fraction': xnew, 'molecular_weight_kda': [calcfr(x) for x in xnew]})
    cal_d.to_csv('cal_predicted.txt', sep='\t', index=False)
    
    # plt.figure(figsize=(6,5))

    # # Plot experimental calibration points (MW in kDa, log10 scale)
    # plt.scatter(fr, 10**mw/1000, color='blue', label='Calibration standards')

    # # Plot fitted line (MW in kDa)
    # plt.plot(xnew, cal_d['MW'], color='red', label=f'Linear fit (R²={lr.score(fr ,mw):.3f})')

    # plt.yscale('log')  # SEC calibration is log-linear
    # plt.xlabel('Fraction number')
    # plt.ylabel('Molecular weight (kDa)')
    # plt.title('SEC Calibration Curve')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return dict(zip(cal_d['fraction'], cal_d['molecular_weight_kda']))


def runner(tmp_, ids, cal, mw, fdr, mode, mrg):
    """
    read folder tmp in directory.
    then loop for each file and create a combined file which contains all files
    creates in the tmp directory
    """
    print(datetime.now())

    dir_ = []
    dir_ = [x[0] for x in os.walk(tmp_) if x[0] is not tmp_]
    # TODO need to keep subcomplexes if they are far away from the predicted
    # molecular weight
    exp_info = pd.read_csv(ids, sep="\t", index_col=0)
    exp_info['cond_rep'] = exp_info['cond'] + "_" + exp_info['repl'].astype(str)
    exp_info = dict(zip(exp_info.index, exp_info['cond_rep']))
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
        if not exp_info.get(base, None):
            continue
        print("Processing sample: {}".format(base))
        mp_feat_norm = os.path.join(smpl, "mp_feat_norm.txt")
        pred_out = os.path.join(smpl, "rf.txt")
        ann = os.path.join(smpl, "cmplx_combined.txt")
        # NB this needed for stoichiometry estimation
        prot = os.path.join(smpl, "transf_matrix.txt")
        raw = os.path.join(smpl, "raw.txt")
        peak = os.path.join(smpl, "peak_list.txt")
        exp = ProphetExperiment(
            feature=mp_feat_norm,
            peaks=peak,
            pred=pred_out,
            prot_matrix=prot,
            raw=raw,
            annotation=ann,
            base=smpl,
            nm=exp_info[base],
            cal=cal,
        )
        if mw != "None":
            mw_df = pd.read_csv(mw, sep="\t")
            mw_df['Gene Names'] = mw_df['Gene Names'].str.split(' ')
            mw_df = mw_df.explode('Gene Names')
            mw_df = dict(zip(list(mw_df["Gene Names"]), list(mw_df["Mass"].astype(int))))
            exp.add_mw(mw_df)
        exp.complex_centric_combine()
        if mrg != 'all':
            # skip fdr
            fdr = -1
        exp.calc_fdr(fdr)
        exp.collapse_hypo(mode=mode)
        exp.peaks_inte_combine()
        allexps.add_exps(exp)
    allexps.multi_collapse()
    allexps.combine_all()
    allexps.calc_subcomplexes()
    final = allexps.protein_centric_combine()
    ## TODO add is_subcomplex column 
    outname = os.path.join(tmp_, "combined.txt")
    final.drop(columns=["cond_rep"], inplace=True, errors='ignore')
    final.to_csv(outname, sep="\t", index=False)
    return True
