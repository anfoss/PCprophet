import sys
import os
import re
import networkx as nx
import pandas as pd
import numpy as np
from scipy import cluster
from scipy.sparse import csr_matrix
import uuid
import scipy.ndimage as image
import scipy.signal as signal_processing
from datetime import datetime

import PCprophet.io_ as io
import PCprophet.mcl as mc


def impute_namean(ls):
    """
    impute 0s in list with value in between if neighbours are values
    assumption is if data is gaussian mean of sequential points is best
    """
    idx = [i for i, j in enumerate(ls) if j == 0]
    for zr in idx:
        if zr == 0 or zr == (len(ls) - 1):
            continue
        elif ls[zr - 1] != 0 and ls[zr + 1] != 0:
            ls[zr] = (ls[zr - 1] + ls[zr + 1]) / 2
        else:
            continue
    return ls


def resample(signal_l, input_fr, output_fr):
    """
    use linear interpolation
    using endpoint=False gets less noise in the resampled
    """
    scale = output_fr / input_fr
    n = round(len(signal_l) * scale)
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),
        np.linspace(0.0, 1.0, len(signal_l), endpoint=False),
        signal_l,
    )
    return resampled_signal


def resize(ls, lower=0, upper=1.0):
    """
    rescale list of values from 1 to 0
    """
    if max(ls) == min(ls):
        return [0] * len(ls)
    else:
        ls_std = [(x - min(ls)) / (max(ls) - min(ls)) for x in ls]
        return [(x * (upper - lower) + lower) for x in ls_std]


def center_row(arr, fr_nr="all", smooth=True, stretch=(True, 72), resc=True):
    """
    Apply normalization and transformation to a single array (row) of a DataFrame.
    """
    key = arr
    if fr_nr != "all":
        key = key[0:(fr_nr)]
    if len([x for x in key if x > 0]) < 2:
        return np.nan
    if smooth:
        key = image.filters.gaussian_filter1d(arr, sigma=1, order=0)
    key = impute_namean(key)
    if stretch[0]:
        key = resample(key, len(key), output_fr=stretch[1])
    if resc:
        key = resize(key)
    return key


def complex_from_clusters(idx_to_gn, clusters):
    ids = "ppi"
    data = [
        [str(i+1), f"{ids}_{i+1}", ";".join(str(idx_to_gn[x]) for x in cmplx)]
        for i, cmplx in enumerate(clusters)
    ]
    return pd.DataFrame(data, columns=["complex_id", "complex_name", "subunits_gene_name"])


def rec_mcl(path):
    df = pd.read_csv(path, sep="\t")
    G = nx.from_pandas_edgelist(df, source="protein1", target="protein2")
    nodelist = list(G.nodes())

    # need to pass the order or csr matrix is random
    matrix = csr_matrix(nx.to_scipy_sparse_array(G, nodelist=nodelist))

    #matrix is CSR format
    #https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.to_scipy_sparse_array.html
    # TODO add weights
    result = mc.run_mcl(matrix)
    clusters = mc.get_clusters(result)
    # optimize and re_run
    opt = mc.run_mcl(matrix, inflation=optimize_mcl(matrix, result, clusters))
    clusters = mc.get_clusters(opt)
    idx_to_gn = dict(enumerate(nodelist))
    df = complex_from_clusters(idx_to_gn, clusters)
    df.to_csv(io.resource_path("ppi_db.txt"), sep='\t', index=False)


def optimize_mcl(matrix, results, clusters):
    newmax = 0
    infl = 0
    for inflation in [i / 10 for i in range(15, 26)]:
        result = mc.run_mcl(matrix, inflation=inflation)
        clusters = mc.get_clusters(result)
        qscore = mc.modularity(matrix=result, clusters=clusters)
        if qscore > newmax:
            infl = inflation
            qscore = newmax
    return infl


def decondense(df, ids):
    """
    Decondense a linkage matrix into a DataFrame with cluster number and members.
    Returns a DataFrame with columns ['cluster_n', 'members'].
    """
    clusters = {}
    rows = cluster.hierarchy.linkage(df)
    lab = dict(zip(range(len(ids)), ids))
    data = []
    for row in range(rows.shape[0]):
        cluster_n = row + len(ids)
        glob1, glob2 = rows[row, 0], rows[row, 1]
        current = []
        for glob in [glob1, glob2]:
            if glob > (len(ids) - 1):
                current += clusters[glob]
            else:
                current.append(lab[int(glob)])
        clusters[cluster_n] = current
        data.append({'complex_id': cluster_n, 'members': current})
    clst_df = pd.DataFrame(data)
    clst_df['members'] = clst_df['members'].apply(lambda x: x if isinstance(x, list) else [])
    return clst_df.explode('members', ignore_index=True)


def split_peaks(prot_arr, skp=0, width=4):
    """
    Split peaks in prot_arr and return a DataFrame.
    
    Each row is a cleaned array corresponding to a peak.
    The index is 'pr_0', 'pr_1', etc.
    """
    # hardcoded parameters for peak detection
    peaks = signal_processing.find_peaks(prot_arr.values, width=width, prominence=0.3, distance=5)
    left_bases = peaks[1]["left_bases"]
    right_bases = peaks[1]["right_bases"]
    fr_peak = peaks[0]

    rows = {}
    
    # If no or only one peak, return the full array under the original name
    if len(fr_peak) < 2:
        return pd.DataFrame([prot_arr], index=[prot_arr.name], columns=prot_arr.index)
    
    cleaned = []
    name  = []
    for idx, pk in enumerate(fr_peak):
        if pk < 6 or pk > 69:
            continue
        name.append([f"{x}_{idx}" for x in prot_arr.name])
        cleaned.append(fill_zeroes(prot_arr.values, pk, left_bases[idx], right_bases[idx]))

    
    df =  pd.DataFrame(cleaned, columns=prot_arr.index)
    df[['protein_id', 'gene_name']] = name
    return df.set_index(['protein_id', 'gene_name'], drop=True)


def fill_zeroes(prot, pk, left_base, right_base):
    """
    Zero out values outside the peak window and regions where signal rises after peak.
    Returns an array of the same shape as the input.
    """
    arr = prot.copy()
    out = np.zeros_like(arr)

    # Limit processing to peak window only
    window = arr[left_base:right_base].copy()
    pk_local = pk - left_base  # peak index relative to window

    # Right side (after peak)
    for i in range(pk_local, len(window) - 1):
        if window[i] < window[i + 1]:
            window[i + 1:] = 0
            break

    # Left side (before peak)
    for i in range(pk_local, 0, -1):
        if window[i] < window[i - 1]:
            window[:i - 1] = 0
            break

    out[left_base:right_base] = window
    return out


def collapse_prot(pr_df, max_size=20):
    peaks_df = [split_peaks(row, row.name) for _, row in pr_df.iterrows()]
    peaks_df = pd.concat(peaks_df)     
    hypo = decondense(peaks_df, list(peaks_df.index))
    hypo['protein_id'] = [x[0] for x in hypo['members']]
    hypo['gene_name'] = [x[1] for x in hypo['members']]
    hypo.drop(columns=['members'], inplace=True)
    subunits = hypo['complex_id'].value_counts()
    hypo = hypo[hypo['complex_id'].isin(subunits[subunits <= max_size].index)]
    # remove single proteins not sure there are though
    hypo = hypo[hypo['complex_id'].isin(subunits[subunits > 1].index)]

    dd = set(hypo['complex_id'].unique())
    dd = dict(zip(dd, ["hypo_" + str(uuid.uuid4()) for x in dd]))
    hypo['complex_id'] = hypo['complex_id'].map(dd)
    return hypo, peaks_df

#TODO check if this keeps reported first or hypothesis first.
def dedup_complexes(df, max_size=30, min_size=2):
    """
    # Remove duplicate complexes based on identical sets of proteins
    # Step 1: For each complex, get the set of proteins (IDs)
    # Step 2: Find unique sets and assign a canonical complex_id
    # Step 3: Map all complexes with identical protein sets to the canonical complex_id

    # Step 4: Drop duplicates based on canonical_complex_id and protein protein_id
    """
    protein_sets = (
    df.groupby('complex_id')['protein_id']
    .apply(lambda ids: frozenset(ids))
    .reset_index()
    )
    merged = (
        protein_sets
        .groupby('protein_id')['complex_id']
        .apply(lambda x: ','.join(sorted(x)))
        .to_dict()
    )
    id_to_merged = {}
    for complex_set in protein_sets.groupby('protein_id')['complex_id']:
        complexes = list(complex_set[1])
        merged_id = ','.join(sorted(complexes))
        for cid in complexes:
            id_to_merged[cid] = merged_id
    df['complex_id'] = df['complex_id'].map(id_to_merged)
    df.drop_duplicates(subset=['complex_id', 'protein_id'], inplace=True)
    complex_sizes = df.groupby('complex_id')['protein_id'].nunique()
    return df[df['complex_id'].isin(complex_sizes[(complex_sizes >= min_size) & (complex_sizes < max_size)].index)]



def runner(infile, db, is_ppi, hypothesis):
    # create subfolder tmp/infile
    print(datetime.now())

    base = io.file2folder(infile, prefix="./tmp/")        
    if not os.path.isdir(base):
        os.makedirs(base)    
    prot = pd.read_csv(infile, sep='\t')
    prot['gene_name'] = prot['gene_name'].apply(lambda x: str.upper(str(x)))
    prot = prot.set_index(['gene_name', 'protein_id'])
    print("Mapping {} to {}".format(infile, db))
    prot = prot.fillna(0)
    #padding preferred to 72. Need to pad to end until 72 so fractions are kept
    #the same number and avoid to distort profiles?
    prot_notnorm = prot.apply(lambda row: center_row(row.values, stretch=(True, 72), smooth=False, resc=False), axis=1)
    # remove rows with NaN values
    prot_notnorm = prot_notnorm.dropna()
    prot_notnorm = prot_notnorm.apply(pd.Series)
    prot_notnorm.to_csv(os.path.join(base, "raw.txt"), sep="\t")
    
    
    # perform normalization
    prot_norm = prot.apply(lambda row: center_row(row.values, stretch=(True, 72)), axis=1)
    prot_norm = prot_norm.dropna()
    prot_norm = prot_norm.apply(pd.Series)

    prot_norm.index.name = "protein_id"
    # create tmp folder and subfolder with name
    # write transf matrix    
    prot_norm.to_csv(os.path.join(base, "transf_matrix.txt"), sep="\t", encoding="utf-8")
    
    
    #### reported complexes
    if is_ppi == "True":
        ppi_path = io.resource_path("ppi_db.txt")
        if not os.path.exists(ppi_path):
            print('PPI network detected, performing network clustering')
            rec_mcl(db)

        db = ppi_path
        print('Generated complex database from PPI network')
    db = pd.read_csv(db, sep='\t')
    
    ## need to put also gene_name in uppercase
    db['complex_id'] = db['complex_name'].astype(str) + "_" + db['complex_id'].astype(str)
    db['subunits_gene_name'] = db['subunits_gene_name'].apply(lambda x: [str.upper(y) for y in x.split(';')])
    db = db.explode('subunits_gene_name')
    db = db[['complex_id', 'subunits_gene_name']]

    # Print number of reported complexes available
    num_available = db['complex_id'].nunique()
    print(f"{num_available} complexes available")

    prot_norm = prot_norm.reset_index()

    ## only keep complexes with more than 30% completeness
    ## TODO add as parameter
    prc = 0.3

    gn = set(prot_norm['gene_name'])

    # Group by complex_id and compute completeness
    cmplt = (
        db.groupby('complex_id')['subunits_gene_name']
        .apply(lambda subunits: sum(g in gn for g in set(subunits)) / len(set(subunits)))
    )
    db['completeness'] = db['complex_id'].map(cmplt.to_dict())

    cmplt = cmplt[cmplt > prc].index
    db = db[db['complex_id'].isin(cmplt)]
    # db = db.rename(columns={'subunits_gene_name': 'gene_name'})
    db_prot = pd.merge(db, prot_norm, right_on='gene_name', left_on='subunits_gene_name', how='inner').drop(columns=['gene_name'])
    if is_ppi == 'True':
        max_size=30
    else:
        max_size=1000
    db_prot = dedup_complexes(db_prot, max_size=max_size, min_size=2)
    db_prot.to_csv(os.path.join(base, "ann_cmplx.txt"), sep="\t", index=False)
    # Print number of mapped complexes
    num_mapped = db_prot['complex_id'].nunique()
    print(f"Number of complexes mapped: {num_mapped}")
    
    
    #### hypothesis generation
    if hypothesis == "all":
        print("Generating hypothesis for {}".format(infile))
        hypo, df_s = collapse_prot(prot_norm.set_index(['protein_id', 'gene_name'], drop=True), max_size=20)
        df_s.to_csv(os.path.join(base, "splitted_transf.txt"), sep="\t")
        base = io.file2folder(infile, prefix="./tmp/")
        hypo_prot = pd.merge(hypo, prot_norm, on=['gene_name', 'protein_id'], how='inner')
        hypo_prot.to_csv(os.path.join(base, "hypo.txt"), sep="\t", index=False)
        hypo_prot = dedup_complexes(hypo_prot, max_size=20, min_size=2)
        hypo_prot = hypo_prot.rename(columns={'gene_name': 'subunits_gene_name'})
        print(f"Number of hypothesis generated: {hypo_prot['complex_id'].nunique()}")
        merged = pd.concat([db_prot, hypo_prot])
        merged = dedup_complexes(merged, max_size=1000, min_size=2)
        print(f"Total number of complexes (reported + hypothesis): {merged['complex_id'].nunique()}")
        merged.to_csv(os.path.join(base, "cmplx_combined.txt"), sep="\t", index=False)
    else:
        db_prot.to_csv(os.path.join(base, "cmplx_combined.txt"), sep="\t", index=False)
    return True


if __name__ == "__main__":
    main()
