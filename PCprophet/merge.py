import os
import sys
import pandas as pd

import PCProphet.io_ as io


def split_delim(dels):
    x_ = dels.split("#")
    x_.sort()
    return "#".join(x_)


def combined_hyp(base):
    # base = io.resource_path(base)
    hypo = pd.read_csv(os.path.join(base, 'hypo.txt'), sep="\t")
    hypo['ANN'] = 0
    hypo['CMPLT'] = 0
    hypo = hypo[['ID', 'CMPLT', 'MB', 'FT', 'ANN']]
    cor = pd.read_csv(os.path.join(base, 'ann_cmplx.txt'), sep="\t")
    cor['ANN'] = 1
    assert list(hypo) == list(cor)
    combined = cor.append(hypo, ignore_index=True)
    # now we fix the duplicate entry
    # this is the issue, makes no sense now to do this
    combined['MB'].apply(lambda x: split_delim(x))
    combined = combined.groupby('MB').agg({'CMPLT': 'first', 'ID': '#'.join, 'FT': 'first', 'ANN': 'first'}).reset_index()
    combined = combined[['ID', 'MB', 'FT', 'ANN', 'CMPLT']]
    return combined


def corum(base):
    # base = io.resource_path(base)
    cor = pd.read_csv(os.path.join(base, 'ann_cmplx.txt'), sep="\t")
    cor['ANN'] = 1
    combined = cor[['ID', 'MB', 'FT', 'ANN', 'CMPLT']]
    return combined

def runner(base, mergemode):
    """
    get both hypothesis and annotated complexes and merge them in single file
    before prediction
    argv[1] = tmp/filename
    """
    # need to reformat both
    combined = pd.DataFrame()
    if mergemode == 'all':
        combined = combined_hyp(base)
        print('combining hypothesis and corum for ' + base)
    elif mergemode == 'reference':
        combined = corum(base)
        print('using only corum annotated complexes for ' + base)
    outpath = os.path.join(base, 'cmplx_combined.txt')
    combined.to_csv(outpath, sep='\t', index=False)
    return True
