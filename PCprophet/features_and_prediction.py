import re
import sys
import os
import numpy as np
import scipy.signal as signal
import pandas as pd
import networkx as nx
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter
import scipy.signal as signal_processing
from dask import dataframe as dd
from dask import bag as db
from dask.diagnostics import ProgressBar
import statistics as stat
import itertools
import joblib
import multiprocessing
import itertools
import pickle
from datetime import datetime

import PCprophet.parse_go as go


np.seterr(all="ignore")
# silence the division by 0 in the correlation calc
mute = np.testing.suppress_warnings()
mute.filter(RuntimeWarning)
mute.filter(module=np.ma.core)


class ProteinProfile(object):
    """
    ProteinProfile represents a protein's intensity profile and provides methods to analyze its features.

    Attributes:
        acc (str): The accession identifier for the protein.
        inten (np.ndarray): The intensity values for the protein profile, converted to a NumPy array of floats.
        peaks (list): List of detected peak indices in the intensity profile.

    Methods:
        __init__(acc, inten):
            Initializes the ProteinProfile with an accession and intensity values.

        get_inte():
            Returns the intensity values as a NumPy array.

        get_acc():
            Returns the accession identifier.

        get_peaks():
            Returns the list of detected peak indices.

        calc_peaks():
            Detects peaks in the intensity profile using signal processing and updates the peaks attribute.
    """


    def __init__(self, acc, inten):
        super(ProteinProfile, self).__init__()
        self.acc = acc
        self.inten = np.array([float(x) for x in inten])
        self.peaks = []

    def get_inte(self):
        return self.inten

    def get_acc(self):
        return self.acc

    def get_peaks(self):
        return self.peaks

    def calc_peaks(self):
        self.peaks, _ =  signal_processing.find_peaks(self.inten, width=4, prominence=0.3)
        # avoid breakage due to float
        self.peaks = list(map(int, self.peaks))


class ComplexProfile(object):
    """
    ComplexProfile represents a protein complex and provides methods to compute and store various features for its members.

    Attributes:
        name (str): Name of the complex.
        goscore (list): List to store GO scores.
        members (list): List of protein members (expected to be ProteinProfile class).
        pks (dict): Dictionary mapping member accession to peak information.
        width (float or dict): Width of the complex or its members.
        shifts (list): List of shift values between member peaks.
        cor (list or np.ndarray): List or array of correlation values between member intensity profiles.
        diff (list or np.ndarray): List or array of difference values between member intensity profiles.
        pks_ali (dict): Dictionary mapping member accession to aligned peak
        positions.
        self._peaks_aligned (bool): Skip calculation of alignment peaks

    Methods:
        __init__(name):
            Initializes a ComplexProfile instance with the given name.
    
        test_complex():
            Checks if the complex has a valid number of members (between 2 and 100).

        add_member(prot):
            Adds a protein member to the complex.

        get_members():
            Returns a list of accession numbers for all members.

        get_name():
            Returns the name of the complex.

        create_matrix():
            Creates a 2D numpy array of intensity profiles for all members.

        get_complex_peaks():
            Returns formatted lists of member accessions, complex IDs, peak positions, and selections.

        calc_go_score(goobj, gaf):
            Calculates and stores the GO score for the complex using provided GO object and annotation file.

        calc_corr(pairs, W=10):
            Calculates and stores the rolling correlation between two member intensity profiles using a sliding window.

        align_peaks():
            Aligns the peaks of all protein members and updates the aligned peak dictionary.

        pairwise():
            Performs pairwise comparisons between all members to compute correlations, differences, and shifts.

        calc_shift(ids):
            Calculates and stores the shift between the aligned peaks of two members.

        calc_diff(p1, p2):
            Calculates and stores the difference between the intensity profiles of two members.

        calc_width():
            Calculates and stores the average full width at half maximum (FWHM) of member peaks.

        create_row():
            Returns a tuple containing all computed features and identifiers for output or further analysis.
    """


    def __init__(self, name):
        super(ComplexProfile, self).__init__()
        self.name = name
        self.goscore = []
        self.members = []
        self.pks = {}
        self.width = {}
        self.shifts = []
        self.cor = []
        self.diff = []
        self.pks_ali = []
        self._peaks_aligned = False

    def test_complex(self):
        if len(self.members) < 2 or len(self.members) > 100:
            return False
        else:
            return True

    def add_member(self, prot):
        self.members.append(prot)

    def get_members(self):
        return [x.get_acc() for x in self.members]

    def get_name(self):
        return self.name

    def create_matrix(self):
        """
        create numpy 2d array for vectorization
        """
        arr = [x.get_inte() for x in self.members]
        return np.array(arr)

    def get_complex_peaks(self):
        """
        
        """
        rows = []
        for acc, value in self.pks.items():
            raw_peaks, selected = value.split("\t")
            rows.append({
                "member": acc,
                "complex_id": self.get_name(),
                "peaks": raw_peaks,
                "selected_peak": selected
            })
        return rows


    def calc_go_score(self, goobj, gaf):
        self.score = go.combine_all(
            goobj,
            gaf,
            np.array(self.get_members()),
        )

    # TODO double check the W here compared to training model PCprophet
    def calc_corr(self, pairs, W=10):
        """
        Vectorized rolling correlation between two vectors with sliding window of size W
        """
        a, b = pairs[0].get_inte(), pairs[1].get_inte()

        a_win = sliding_window_view(a, W).copy()
        b_win = sliding_window_view(b, W).copy()

        a_win -= a_win.mean(axis=1, keepdims=True)
        b_win -= b_win.mean(axis=1, keepdims=True)

        num = np.einsum('ij,ij->i', a_win, b_win)
        denom = np.sqrt(
            np.einsum('ij,ij->i', a_win, a_win) *
            np.einsum('ij,ij->i', b_win, b_win)
        )

        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.true_divide(num, denom)
            corr[denom == 0] = np.nan

        pad_len = len(a) - len(corr)
        padded = np.hstack([corr, np.full(pad_len, np.nan)])

        self.cor.append(padded)


    def align_peaks(self):
        """
        align all protein peaks
        """
        if self._peaks_aligned:
            return True if self.pks_ali else None  # already done

        pk = [prot.get_peaks() for prot in self.members]
        idx_missing = [i for i, j in enumerate(pk) if not j]
        nan_members = [self.get_members()[i] for i in idx_missing]
        pres = [x for x in pk if x]
        mb_pres = [x for x in self.get_members() if x not in nan_members]
        if nan_members == self.get_members():
            self.pks_ali = dict(zip(nan_members, [np.nan] * len(nan_members)))
            self._peaks_aligned = True
            return None
        else:
            ali_pk = alligner(pres)
            md = round(stat.median(ali_pk))
            # missing values gets the median of aligned peaks
            self.pks_ali = dict(zip(nan_members, [md] * len(nan_members)))
            self.pks_ali.update(dict(zip(mb_pres, ali_pk)))
            for k in self.members:
                if k.get_peaks():
                    _ = "#".join(map(str, k.get_peaks()))
                else:
                    _ = str(self.pks_ali[k.get_acc()])
                pks = _ + "\t" + str(self.pks_ali[k.get_acc()])
                self.pks[k.get_acc()] = pks
            self._peaks_aligned = True
            return True

    def pairwise(self):
        """
        performs pairwise comparison
        """
        for pairs in itertools.combinations(np.array(self.members), 2):
            self.calc_corr(pairs)
            self.calc_diff(*pairs)
            self.calc_shift([x.get_acc() for x in pairs])
        # now need to average
        self.cor = np.mean(self.cor, axis=0)
        self.diff = np.mean(self.diff, axis=0)
        self.shifts = np.mean(self.shifts)

    def calc_shift(self, ids):
        self.shifts.append(abs(self.pks_ali[ids[0]] - self.pks_ali[ids[1]]))

    def calc_diff(self, p1, p2):
        self.diff.append(abs(p1.get_inte() - p2.get_inte()))

    def calc_width(self):
        q = 5
        width = []
        for prot in self.members:
            peak = int(self.pks_ali[prot.get_acc()])
            prot_peak = prot.get_inte()[(peak - q) : (peak + q)]
            prot_fwhm = fwhm(list(prot_peak))
            width.append(prot_fwhm)
        self.width = np.mean(width)

    def create_row(self):
        """
        Flatten all features into one row of primitive types.
        """
        row = {
            "complex_id": self.get_name(),
            "members": "#".join(self.get_members()),
            "SHFT": self.shifts,
            "W": self.width,
            "SC_CC": self.score[0],
            "SC_MF": self.score[1],
            "SC_BP": self.score[2],
            "TOTS": self.score[3],
        }
        row.update({f"COR_{i}": v for i, v in enumerate(self.cor)})
        row.update({f"DIF_{i}": v for i, v in enumerate(self.diff)})

        return row


def fwhm(y, frac=2):
    """
    calculate full width half max of peak within two fractions
    """
    if not y:
        return np.nan
    y = np.array(y)
    x = [x for x in range(1, (len(y) + 1))]
    d = y - (max(y) / frac)
    indexes = np.where(d > 0)[0]
    try:
        return abs(x[indexes[-1]] - x[indexes[0]])
    except IndexError as e:
        return np.nan


def add_top(result, item):
    """Inserts item into list of results"""
    length = len(result)
    index = 0
    # if less than lenght and better diff
    while index < length and result[index][1] < item[1]:
        index += 1
    result.insert(index, item)


def minimize(solution):
    """Returns total difference of solution passed"""
    length = len(solution)
    result = 0
    for index, number1 in enumerate(solution):
        for nr_2_indx in range(index + 1, length):
            result += abs(number1 - solution[nr_2_indx])
    return result


def min_sd(aoa):
    rf_pk = []
    for v in aoa:
        rf_pk.append([x for x in v if x is not None])
    ln = max([len(x) for x in rf_pk])
    rf_pk2 = [x[:ln] for x in rf_pk]
    for short in rf_pk2:
        while len(short) < ln:
            try:
                short.append(short[-1])
            except IndexError as e:
                break
    pkn = pd.DataFrame(rf_pk2)
    # now all peaks detected are alligned rowise
    # calc standard deviation and take index of min sd
    sd = (pkn.apply(lambda col: np.std(col, ddof=1), axis=0)).tolist()
    try:
        sd = sd.index(min(sd))
    except ValueError as e:
        return None
    indx = []
    # for each protein append index of peak
    # input is protA [peak, peak ,peak]
    # indx out is [protA=> peak, protB => peak, protC => peak]
    # order is same because we append from same array
    for mx_indx in rf_pk2:
        try:
            indx.append(mx_indx[sd])
        # if no peak in mx_indx append none
        except IndexError as e:
            indx.append(None)
    return indx


def shortest_path(aoa, max_trial=5000):
    elements = len(aoa)
    result = [[[x], 0] for x in aoa[0]]
    trial = 1
    while True:
        if trial == max_trial:
            return None
        trial += 1
        sol = result.pop(0)
        # print(sol)
        # Return the top item if it is complete
        if len(sol[0]) == elements:
            return sol[0]
            # Make new solutions with top item
        for peak in aoa[len(sol[0])]:
            new_pk = [sol[0].copy(), 0]
            new_pk[0].append(peak)
            new_pk[1] = minimize(new_pk[0])
            add_top(result, new_pk)


def alligner(aoa):
    """Finds closest points of a list of lists"""
    # one of arrays is empty
    for x in aoa:
        if not x:
            return None
    # there is the same nr in all array no need to do anything
    candidate = set.intersection(*map(set, aoa))
    if candidate:
        # returns intersect
        return [max(list(candidate))] * len(aoa)
    else:
        pks = shortest_path(aoa)
        if pks:
            return pks
        else:
            pks = min_sd(aoa)
            return pks



def convert_to_complex_profile(subdf):
    tmp = ComplexProfile(subdf['complex_id'].iloc[0])
    for _, row in subdf.iterrows():
        acc = row[['subunits_gene_name', 'protein_id']]
        # for duplicates but not really needed
        v_cols = [col for col in subdf.columns if col not in ['complex_id', 'subunits_gene_name', 'protein_id']]
        intensities = row[v_cols].tolist()
        ## NOTE this is important, needs to be GN otherwise no match in GAF and OBO
        protein = ProteinProfile(acc.iloc[0], intensities)
        protein.calc_peaks()
        tmp.add_member(protein)
    tmp.test_complex()
    return tmp


def gen_feat(s, goobj, gaf):
    """
    receive a single row and generate feature calc
    """
    cmplx = convert_to_complex_profile(s)
    if cmplx.test_complex() and cmplx.align_peaks():
        cmplx.calc_go_score(goobj, gaf)
        cmplx.calc_width()
        cmplx.pairwise()
        return cmplx.create_row()
    else:
        return None

def gen_peaks(s):
    cmplx = convert_to_complex_profile(s)
    if cmplx.test_complex() and cmplx.align_peaks():
        return cmplx.get_complex_peaks()
    else:
        return None


def runner(base, go_obo, tsp_go, model):
    """
    generate all features from the mapped complexes file
    base = config[GLOBAL][TEMP]filename
    """
    print(datetime.now())

    go_tree = nx.read_graphml(go_obo)
    with open(tsp_go, "rb") as f:
        gaf = pickle.load(f)
    cmplx_file = os.path.join(base, "cmplx_combined.txt")
    df = pd.read_csv(cmplx_file, sep="\t")

    n_partitions = multiprocessing.cpu_count() * 8

    # change to dask delayed
    grouped = df.groupby("complex_id")
    records = [(cid, group) for cid, group in grouped]
    bag = db.from_sequence(records, npartitions=n_partitions)
    print("Calculating features for {}".format(cmplx_file))
    with ProgressBar():
        feats = (
            bag
            .map(lambda x: gen_feat(x[1], go_tree, gaf))
            .filter(lambda x: x is not None)
            .compute()
        )
    
    feats = pd.DataFrame(feats)
    pks = (
        bag
        .map(lambda x: gen_peaks(x[1]))
        .filter(lambda x: x is not None)
        .compute()
    )
    pks = list(itertools.chain.from_iterable(pks))
    pks = pd.DataFrame(pks)

    feature_path = os.path.join(base, "mp_feat_norm.txt")
    feats.to_csv(feature_path, sep="\t", index=False)

    peaklist_path = os.path.join(base, "peak_list.txt")
    pks.to_csv(peaklist_path, sep="\t", index=False)
    
    
    ### prediction
    # NOTE 72 length hardcoded
    cor_cols = [f"COR_{i}" for i in range(72)]
    dif_cols = [f"DIF_{i}" for i in range(72)]
    ordered_cols = ["SHFT", "W"] + cor_cols + dif_cols
    feat_num = feats[ordered_cols].copy().apply(pd.to_numeric)
    feat_num.replace([np.nan, "nan", "na", ""], 0, inplace=True)

    print('Predicting interactions for {}'.format(cmplx_file))
    clf = joblib.load(model)
    prob = clf.predict_proba(feat_num.values)
    pred = clf.predict(feat_num.values)
    df = pd.DataFrame({
        "complex_id": feats['complex_id'],               
        "rf_probability": prob[:, 1],
        "is_complex": np.where(pred == 1, "Yes", "No")
    })[["complex_id", "rf_probability", "is_complex"]]
    df.to_csv(os.path.join(base, "rf.txt"), sep="\t", index=False)
 
    return True
