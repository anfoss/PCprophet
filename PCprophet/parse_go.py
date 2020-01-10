import re
import itertools
import networkx as nx

import PCprophet.io_ as io
import PCprophet.stats_ as st


class GoGraph(nx.DiGraph):
    """Directed acyclic graph of Gene Ontology
    Attributes:
        alt_ids(dict): alternative IDs dictionary
        descriptors(set): flags and tokens that indicates the graph is
            specialized for some kind of analyses
        lower_bounds(collections.Counter):
            Pre-calculated lower bound count (Number of descendants + 1).
            Information content calculation requires precalc lower bounds.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alt_ids = {}  # Alternative IDs
        self.descriptors = set()
        self.lower_bounds = None
        # self.reversed = self.reverse(copy=False)


def parse_block(lines):
    """Parse a Term block
    """
    term = {"alt_id": [], "relationship": []}
    splitkv = re.compile(r"(^[a-zA-Z_]+): (.+)")
    for line in lines:
        m = re.search(splitkv, line)
        # assert m, f"unexpected line: {line}"
        key = m.group(1)
        value = m.group(2)
        if key in ["id", "name", "namespace", "is_obsolete"]:
            term[key] = value
        elif key == "alt_id":
            term["alt_id"].append(value)
        elif key == "is_a":
            goid = value.split("!")[0].strip()
            term["relationship"].append({"type": "is_a", "id": goid})
        elif key == "relationship":
            typedef, goid = value.split("!")[0].strip().split(" ")
            term["relationship"].append({"type": typedef, "id": goid})
    return term


def blocks_iter(lines):
    """Iterate Term (and Typedef) blocks
    """
    type_ = None
    content = []
    termdef = re.compile(r"^\[([a-zA-Z_]+?)\]$")
    for line in lines:
        m = re.search(termdef, line)
        if m:
            if type_ is not None and content:
                yield {"type": type_, "content": content[:]}
            type_ = m.group(1)
            content.clear()
        elif line.rstrip():
            content.append(line.rstrip())
    if content:
        yield {"type": type_, "content": content[:]}


def from_obo_lines(lines, ignore_obsolete=True):
    lines_iter = iter(lines)

    # Build graph
    G = GoGraph()
    alt_ids = set()

    # Term blocks
    for tb in blocks_iter(lines_iter):
        if tb["type"] != "Term":
            continue
        term = parse_block(tb["content"])

        # Ignore obsolete term
        obso = term.get("is_obsolete") == "true"
        if obso and ignore_obsolete:
            continue

        # Alternative ID mapping
        alt_ids |= set(term["alt_id"])
        for alt_id in term["alt_id"]:
            G.alt_ids[alt_id] = term["id"]

        # Add node
        attr = {
            "name": term["name"],
            "namespace": term["namespace"],
            "is_obsolete": obso,
        }
        G.add_node(term["id"], **attr)
        for rel in term["relationship"]:
            G.add_edge(rel["id"], term["id"], type=rel["type"])

    # Check
    assert not (set(G) & alt_ids), "Inconsistent alternative IDs"
    assert len(G) >= 2, "The graph size is too small"
    assert G.number_of_edges(), "The graph has no edges"
    return G


def from_obo(pathlike, **kwargs):
    with open(pathlike, "rt") as f:
        G = from_obo_lines(f, **kwargs)
    return G


def read_gaf_out(go_path):
    """
    read gaf file and create a hash of hash
    gn => c
       => mf
       => bp
    """
    out = io.makedeephash()
    header = []
    temp = {}
    for line in open(go_path, mode="r"):
        line = line.rstrip("\n")
        if line.startswith(str("ID") + "\t"):
            header = re.split(r"\t+", line)
        else:
            things = re.split(r"\t+", line)
            temp = dict(zip(header, things))
        if len(temp.keys()) > 0:
            pr = str.upper(temp["GN"])
            for k in temp.keys():
                # if the key is the same
                if out[pr][k] and k is not "ID" or "GN":
                    out[pr][k] = ";".join([str(out[pr][k]), temp[k]])
                elif k is not "ID" or "GN":
                    out[pr][k] = temp[k]
    return out


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
    retrieve the GO gene names term from the
    """
    tmp = []
    try:
        tmp = gaf[gn][go_type].split(";")
    except AttributeError as e:
        tmp.append("NA")
    tmp = list(set(tmp))
    return [x for x in tmp if x is not "NA"]


def scr(G, gaf, id1, id2, go_type):
    """
    score using wang
    """
    t1 = parse_go(id1, gaf, go_type)
    t2 = parse_go(id2, gaf, go_type)
    if t1 and t2:
        x = [(wang(G, x[0], x[1])) for x in list(itertools.product(t1, t2))]
        return st.mean(x)
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
        k = [scr(G, gaf, x[0], x[1], go) for x in list(st.fast_comb(t, 2))]
        out.append(st.mean(k))
    # add to out the mean of the three Ontologies
    # TODO check mean or sum
    out.append(st.mean(out))
    return "\t".join([str(x) for x in out])


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
