#!/usr/bin/env python3

import re
import sys
import os
import numpy as np
from scipy import stats
import pandas as pd

import PCProphet.signal_prc as preproc
import PCProphet.parse_go as go
import PCProphet.io_ as io
import PCProphet.stats_ as st


def add_top(result, item):
    """Inserts item into list of results"""
    length = len(result)
    index = 0
    # if less than lenght and better diff
    while (index < length and result[index][1] < item[1]):
        index += 1
    result.insert(index, item)


def minimize(solution):
    """Returns total difference of solution passed """
    length = len(solution)
    result = 0
    for index, number1 in enumerate(solution):
        for nr_2_indx in range(index + 1, length):
            result += abs(number1 - solution[nr_2_indx])
    return result


def naive_allign(aoa):
    ls = [(idx, v) for idx, sublist in enumerate(aoa) for v in sublist]
    ls = sorted(ls, key=lambda pair: pair[1])
    ids, ls = zip(*ls)
    # if nr of elements in aoa is the same as list length then all list is sol
    if len(set(ids)) == len(ls):
        return ls
    # find median and indexes of all appearence of median
    md = np.median(np.array(ls), axis=0)
    idxs = [idx for idx, v in enumerate(ls) if v == int(md)]
    # now we see the closes element to the median on both sides
    # nr of elements missing to solution
    left = len(aoa) - len(idxs)
    # the possible solutions are idxs +- left on both sides
    inf = list(range(idxs[0] - left, idxs[0]))
    sup = list(range(idxs[-1] + 1, idxs[-1] + left + 1))
    # trim list to have only real values in list
    inf = [x for x in inf if x > - 1]
    sup = [x for x in sup if x < len(ls)]
    # start from last one of inf and first one of sup
    inf_indx = -1
    sup_indx = 0
    # exist inf or sup
    if sup and inf:
        while left:
            if abs(ls[inf[inf_indx]] - md) < abs(ls[sup[sup_indx]] - md):
                # we add the left one
                idxs.append(inf[inf_indx])
                inf_indx -= 1
            else:
                # we add the right one
                idxs.append(sup[sup_indx])
                sup_indx += 1
            left -= 1
    elif sup:
        while left:
            left -= 1
            idxs.append(sup[sup_indx])
            sup_indx += 1
    elif inf:
        while left:
            left -= 1
            idxs.append(inf[inf_indx])
            inf_indx += 1
    pks = []
    for i in idxs:
        # we know in which list to search because ids contains list nr
        indx = aoa[ids[i]].index(ls[i])
        pks.append(aoa[ids[i]][indx])
    return pks


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
    while (True):
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


# wrapper
def mp_cmplx(filename, goobj, gaf, window=10, q=5):
    """
    map complex into 3 vector => cor vectors
    shift peak
    width peak
    window = point for correlation
    cor(A[idx:(idx + window)], B[idx:(idx+window)])
    width = fwhm(A[idx-q:idx+q])
    so q should be 1/2 of window ?
    """
    things, header = [], []
    temp = {}
    to_wrout = []
    pks_out = []
    print('calculating features for ' + filename)
    for line in open(filename, 'r'):
        line = line.rstrip('\n')
        if line.startswith('ID' + '\t'):
            header = re.split(r'\t+', line)
        else:
            things = re.split(r'\t+', line)
            temp = {}
            temp = dict(zip(header, things))
        if temp:
            out = {}
            pks_ = {}
            cmplx = temp['FT'].split('#')
            members = temp['MB'].split('#')
            # now add average go for each members
            score = go.combine_all2(goobj, gaf, members)
            cmplx_2 = []
            try:
                for mb in cmplx:
                    cmplx_2.append([float(x) for x in mb.split(',')])
            except ValueError as e:
                continue
            shft, cor, width, dif = [], [], {}, []
            # calculate all peaks
            pk = []
            cmplx_member = re.split(r'#', temp['MB'])
            tmp = dict(zip(cmplx_member, cmplx_2))
            cmplx_member = sorted(cmplx_member)
            for idx, prot in enumerate(cmplx_2):
                pks = list(preproc.peak_picking(prot)[0])
                # we append index
                pk.append([x for x in pks])
                pks_[members[idx]] = '#'.join([str(x) for x in pks])
                # reformat to allign peaks remove nan and sub with first non na
            # print(temp['ID'])
            # pk = [x if x else [0] for x in pk]
            if pk:
                indx = alligner(pk)
            else:
                continue
            if not indx:
                continue
            for idx, mb in enumerate(cmplx):
                pks_[members[idx]] = "\t".join([pks_[members[idx]],
                                                str(indx[idx])])
            # *pairs = ((idx1,[arr1]),(idx2,[arr2]))
            # substitute 0 with with index peaks
            indx = [int(st.mean(indx)) if x is 0 else int(x) for x in indx]
            for pairs in st.fast_comb(list(enumerate(cmplx_2))):
                # get one tuple at the time
                idx_0 = indx[pairs[0][0]]
                idx_1 = indx[pairs[1][0]]
                shft.append(abs(idx_0 - idx_1))
                pairs0 = preproc.fwhm(pairs[0][1][(idx_0 - q):idx_0 + q])
                width[pairs[0][0]] = pairs0
                pairs1 = preproc.fwhm(pairs[1][1][(idx_1 - q):idx_1 + q])
                width[pairs[1][0]] = pairs1
                # cor and diff array wide
                tmp = []
                tmp_dif = []
                for i, (x, y) in enumerate(zip(*[x[1] for x in pairs])):
                    # this is the difference
                    tmp_dif.append(abs(x - y))
                    try:
                        tmp.append(stats.pearsonr(
                                                  pairs[0][1][i:(i + window)],
                                                  pairs[1][1][i:(i + window)]
                                                  )[0])
                    # end of array x + 5 window = 72
                    except IndexError as e:
                        break
                cor.append(tmp)
                dif.append(tmp_dif)
            df = pd.DataFrame(cor)
            # df are Fraction = cols prots = rows so mean is colwise
            # while maintaining the number of rows to 72
            # TODO use # for consistency
            cmplx_mb = '#'.join(cmplx_member)
            cr = df.apply(lambda col: np.nanmean(col), axis=0)
            tm_df = pd.DataFrame(dif)
            tm_df = tm_df.apply(lambda col: np.nanmean(col), axis=0)
            out['d'] = ','.join(str(x) for x in tm_df.tolist())
            out['c'] = ','.join(str(x) for x in cr.tolist())
            out['s'] = str(np.mean(shft))
            out['w'] = str(np.nanmean(np.array(list(width.values()))))
            out['cb'] = cmplx_mb
            tm = {}
            tm[temp['ID']] = cr.tolist()
            row = [temp['ID'], out['cb'], out['c'], out['s'],
                   out['d'], out['w'], str(score)
                   ]
            to_wrout.append("\t".join(row))
            for k in pks_:
                pks_out.append("\t".join([k, temp['ID'], pks_[k]]))
    return set(to_wrout), pks_out


def runner(base, go_obo, tsp_go):
    """
    generate all features from the mapped complexes file
    base = config[GLOBAL][TEMP]filename
    """
    go_tree = go.from_obo(io.resource_path(go_obo))
    gaf = go.read_gaf_out(io.resource_path(tsp_go))
    # get tmp/filename folder
    cmplx_comb = os.path.join(base, 'cmplx_combined.txt')
    # print(os.path.dirname(os.path.realpath(__file__)))
    wr, pks = mp_cmplx(filename=cmplx_comb, goobj=go_tree, gaf=gaf)
    feature_path = os.path.join(base, 'mp_feat_norm.txt')
    feat_header = ['ID', 'MB', 'COR', 'SHFT', 'DIF',
                   'W', 'SC_CC', 'SC_MF', 'SC_BP', 'TOTS'
                   ]
    io.wrout(wr, feature_path, feat_header)
    peaklist_path = os.path.join(base, 'peak_list.txt')
    # peaklist_path = peaklist_path
    io.wrout(pks, peaklist_path, ['MB', 'ID', 'PKS', 'SEL'])
