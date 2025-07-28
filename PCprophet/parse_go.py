import re
import itertools
import networkx as nx
import itertools 
import numpy as np
from collections import defaultdict


def s_values(G, term):
    wf = dict(zip(("is_a", "part_of"), (0.8, 0.6)))
    sv = {term: 1}
    visited = set()
    level = {term}
    while level:
        visited |= level
        next_level = set()
        for n in level:
            for pred, edge in G.pred[n].items():
                weight = sv[n] * wf.get(edge["type"], 0)
                if pred not in sv:
                    sv[pred] = weight
                else:
                    sv[pred] = max([sv[pred], weight])
                if pred not in visited:
                    next_level.add(pred)
        level = next_level
    return {k: round(v, 3) for k, v in sv.items()}


def wang(G, term1, term2):
    """Semantic similarity based on Wang method
    Args:
        G(GoGraph): GoGraph object
        term1(str): GO term
        term2(str): GO term
        weight_factor(tuple): custom weight factor params
    Returns:
        float - Wang similarity value
    Raises:
        PGSSLookupError: The term was not found in GoGraph
    """
    if term1 not in G or term2 not in G:
        return 0
    sa = s_values(G, term1)
    sb = s_values(G, term2)
    sva = sum(sa.values())
    svb = sum(sb.values())
    common = set(sa.keys()) & set(sb.keys())
    cv = sum(sa[c] + sb[c] for c in common)
    return round(cv / (sva + svb), 3)


def parse_go(gn, gaf, go_type):
    """
    Retrieve GO terms for a gene name from the pickled nested dictionary.
    """
    try:
        return gaf[gn][go_type]
    except KeyError:
        return ["NA"]


def scr(G, gaf, id1, id2, go_type):
    """
    score using wang
    """
    t1 = parse_go(id1, gaf, go_type)
    t2 = parse_go(id2, gaf, go_type)
    if t1 and t2:
        x = [(wang(G, x[0], x[1])) for x in list(itertools.product(t1, t2))]
        return sum(x) / len(x) if len(x) > 0 else 0
    else:
        return 0


# if we can make this one
def combine_all(G, gaf, t):
    """
    permute all of blocks of whatever
    """
    go_type = ["CC", "MF", "BP"]
    out = []
    for go in go_type:
        k = [scr(G, gaf, x[0], x[1], go) for x in list(itertools.combinations(t, 2))]
        out.append(sum(k) / len(k) if len(k) > 0 else 0)
    out.append(sum(out))
    return out


def common_parent(terms, go):
    """
    This function finds the common ancestors in the GO
    tree of the list of terms in the input.
    - input:
        - terms: list of GO IDs
        - go: the GO Tree object
    Taken from 'A Gene Ontology Tutorial in Python - Model Solutions to Exercises'
    by Alex Warwick
    """
    # Find candidates from first
    rec = go[terms[0]]
    candidates = rec.get_all_parents()
    candidates.update({terms[0]})

    # Find intersection with second to nth term
    for term in terms[1:]:
        rec = go[term]
        parents = rec.get_all_parents()
        parents.update({term})
        # Find the intersection with the candidates, and update.
        candidates.intersection_update(parents)
    return candidates
