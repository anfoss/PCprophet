import sys
import os
import re
import networkx as nx

import PCprophet.io_ as io
import PCprophet.mcl as mc
import PCprophet.signal_prc as preproc


# standardize and center methods
def center_arr(hoa, fr_nr="all", norm=True, nat=True, stretch=(True, 72)):
    norm = {}
    for k in hoa:
        key = hoa[k]
        if fr_nr != "all":
            key = key[0:(fr_nr)]
        if len([x for x in key if x > 0]) < 2:
            continue
        key = preproc.gauss_filter(key, sigma=1, order=0)
        key = preproc.impute_namean(key)
        if stretch[0]:
            # input original length wanted length
            key = preproc.resample(key, len(key), output_fr=stretch[1])
        key = preproc.resize(key)
        norm[k] = list(key)
    return norm


def rec_mcl(path):
    g = io.ppi2graph(path)
    matrix = nx.to_scipy_sparse_matrix(g)
    result = mc.run_mcl(matrix)
    clusters = mc.get_clusters(result)
    opt = mc.run_mcl(matrix, inflation=optimize_mcl(matrix, result, clusters))
    clusters = mc.get_clusters(opt)
    node = dict(enumerate(g.node()))
    io.create_db_from_cluster(node, clusters)
    return True


def optimize_mcl(matrix, results, clusters):
    newmax = 0
    infl = 0
    for inflation in [i / 10 for i in range(15, 26)]:
        result = mc.run_mcl(matrix, inflation=inflation)
        clusters = mc.get_clusters(result)
        qscore = mc.modularity(matrix=result, clusters=clusters)
        if qscore > newmax:
            infl = inflation
    return infl


def runner(infile, db, is_ppi, use_fr):
    """
    argv[1] = input name conv2gn out
    argv[2] = db
    argv[3] = is_ppi
    """
    prot = io.read_txt(infile)
    print("mapping " + infile + " to " + db)
    prot = center_arr(prot, fr_nr=use_fr, stretch=(True, 72))
    pr_df = io.create_df(prot)
    pr_df = pr_df.loc[~(pr_df == 0).all(axis=1)]
    pr_df.index.name = "ID"
    header = []
    temp = {}
    out = []
    base = io.file2folder(infile, prefix="./tmp/")
    # create tmp folder and subfolder with name
    if not os.path.isdir(base):
        os.makedirs(base)
    # write transf matrix
    dest = os.path.join(base, "transf_matrix.txt")
    pr_df.to_csv(dest, sep="\t", encoding="utf-8")
    if is_ppi == "True":
        # cluster the ppi db into a database
        rec_mcl(db)
        db = io.resource_path("./ppi_db.txt")
    for line in open(db, "r"):
        line = line.rstrip("\n")
        if line.startswith("ComplexID" + "\t"):
            header = re.split(r"\t+", line)
        else:
            things = re.split(r"\t+", line)
            temp = dict(zip(header, things))
        if temp:
            members = re.split(r";", temp["subunits(Gene name)"])
            members = [str.upper(x) for x in members]
            memb, feat = [], []
            for cmplx in members:
                if prot.get(cmplx, None):
                    feat.append(",".join(str(x) for x in prot[cmplx]))
                    memb.append(cmplx)
            if len(feat) > 1:
                nm = temp["ComplexName"] + "_" + temp["ComplexID"]
                nm = nm.replace('"', "")
                ft_v = "#".join(feat)
                mb_v = "#".join(memb)
                cmplt = float(len(memb)) / float(len(members))
                out.append("\t".join([nm, str(cmplt), mb_v, ft_v]))
    nm = os.path.join(base, "ann_cmplx.txt")
    # nm = io.resource_path(nm)
    io.wrout(out, nm, ["ID", "CMPLT", "MB", "FT"])
