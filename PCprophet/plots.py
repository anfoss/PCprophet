import os
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import pandas as pd
import numpy as np

import PCprophet.stats_ as st


def smart_makefold(path, folder):
    """
    """
    pl_dir = os.path.join(path, folder)
    if not os.path.isdir(pl_dir):
        os.makedirs(pl_dir)
    return pl_dir


def plot_positive(comb, sid, pl_dir):
    """
    create 1 dir per condition and plot each complex across repl
    """
    def rescale_fract(row, sid):
        ids, rep = row["COND"], row["REPL"]
        fr = sid[(sid["cond"] == ids) & (sid["repl"] == rep)]["fr"].values
        return np.array(st.resize_plot(row["INT"], input_fr=72, output_fr=fr))

    def rescale_peak(row, sid):
        ids, rep = row["COND"], row["REPL"]
        fr = sid[(sid["cond"] == ids) & (sid["repl"] == rep)]["fr"].values
        # renormalize between peak picking rescaled to 0 fr-1 for highlight
        return st.renormalize(row["SEL"], (0, 71), (0, fr - 1))

    sa_id = pd.read_csv(sid, sep="\t", index_col=False)
    comb = pd.read_csv(comb, sep="\t", index_col=False)

    # remove columns with single protein ID
    comb = comb[comb["ID"] != comb["CMPLX"]]
    comb["reINT"] = comb.apply(lambda row: rescale_fract(row, sa_id), axis=1)
    comb["reSEL"] = comb.apply(lambda row: rescale_peak(row, sa_id), axis=1)

    # now we make one folder per condition
    conds = [os.path.join(pl_dir, x) for x in list(set(comb["COND"]))]
    [os.mkdir(x) for x in conds if not os.path.isdir(x)]
    comb.groupby(["CMPLX", "COND"], as_index=False).apply(
        lambda df: plot_repl_prof(df, pl_dir, cols=1)
    )
    return True


def plot_repl_prof(filt, fold, cols):
    """
    plot profile of protein across groups
    """
    def plot_single(axrow, filt, v, csfont):
        filt2 = filt[filt['REPL'] == v]
        # pk = np.median(filt2["reSEL"].values)
        fractions = [int(x) for x in range(1, len(filt2["reINT"].iloc[0]) + 1)]
        axrow.grid(color="grey", linestyle="--", linewidth=0.25, alpha=0.5)
        for index, row in filt2.iterrows():
            axrow.plot(
                fractions, row["reINT"], "-", lw=1, label=str(row["ID"]),
            )
        # remove highlight for peak area of positive
        # if np.median(filt2["P"].values >= 0.5):
        #     axrow.axvspan(pk - 3, pk + 3, color="grey", alpha=0.2)
        axrow.set_ylabel("Rescaled intensity", fontsize=9, **csfont)

    csfont = {"fontname": "sans-serif"}
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["grid.color"] = "k"
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5
    repl = set(filt['REPL'])
    fig, axes = plt.subplots(len(set(repl)),
                             figsize=(9, 9),
                             facecolor="white",
                             sharex=True,
                             )
    if len(repl) == 1:
        axes = [axes]
    for i, row in enumerate(axes):
        plot_single(row, filt, list(repl)[i], csfont)
    # call tight layout before legend
    handles, labels = axes[-1].get_legend_handles_labels()
    lgd = fig.legend(
                    handles,
                    labels,
                    # bbox_to_anchor=(0.5,-0.02),
                    loc='lower center',
                    ncol=6
                    )
    nm = filt["CMPLX"].values[0]
    ids = nm
    fig.suptitle("\n".join(nm.split("#")), fontsize=12, **csfont)
    plt.xlabel("Rescaled fraction (arb. unit)", fontsize=9, **csfont)
    if "/" in ids:
        ids = ids.replace("/", " ")
    cnd = list(set(filt['COND']))
    plotname = os.path.join(fold, cnd[0], "%s.pdf" % str(ids))
    try:
        fig.savefig(
                    plotname,
                    dpi=800,
                    bbox_inches="tight",
                    )
    except OSError as exc:
        # catch name too long
        if exc.errno == 63:
            ids = ids.split("#")[0]
            plotname = os.path.join(fold, cnd[0], "%s.pdf" % str(ids))
            fig.savefig(
                        plotname,
                        dpi=800,
                        bbox_inches="tight",
                        )
        else:
            raise exc
    plt.close()
    return True


def plot_network(outf, ppi="./PPIReport.txt"):
    """
    plot network
    """

    def filter_report(df):
        """
        take a df with ppi, sort by reported and not
        and keep the first one (i.e reported)
        in case that is present
        """
        df = df.sort_values(by="Reported", ascending=True)
        df.drop_duplicates(
            subset=["ProteinA", "ProteinB", "Replicate", "Condition"],
            keep="last",
            inplace=True,
        )
        return df

    df = pd.read_csv(os.path.join(outf, ppi), sep="\t")
    df = df.sort_values(by="Reported", ascending=True)
    df.drop_duplicates(
        subset=["ProteinA", "ProteinB", "Replicate", "Condition"],
        keep="last",
        inplace=True,
    )
    df = filter_report(df)
    counts = df.groupby(["ProteinA", "ProteinB"]).size().reset_index()
    df = pd.merge(df, counts, on=["ProteinA", "ProteinB"])

    # now drop duplicates per replicate
    df.drop_duplicates(subset=["ProteinA", "ProteinB"], keep="first", inplace=True)

    # need to drop the duplicates
    df_gr = nx.Graph()
    df_gr.add_edges_from(zip(df["ProteinA"], df["ProteinB"]))
    probs = None
    try:
        probs = pd.read_csv("DifferentialProteinReport.txt", sep="\t")
        probs = dict(
            zip(probs["Gene name"], probs["ProbabilityDifferentialRegulation"])
        )
    except FileNotFoundError as e:
        pass
    if probs:
        probs = [probs.get(x, 0) for x in df_gr.nodes]
        # now rescale from 1 to 8
        probs = [st.renormalize(x, (0, 1), (2, 6)) for x in probs]
    else:
        probs = [4] * len(df_gr.nodes)
    nx.write_graphml(df_gr, "ppi_network.GraphML")
    df_gr = ig.Graph.Read_GraphML("ppi_network.GraphML")

    # calculate network stats
    community = df_gr.community_multilevel()
    comm = max(community.membership)
    # initialize color palette
    cols = ig.ClusterColoringPalette(comm + 1)
    clr = [cols.get(x) for x in community.membership]

    #  create edge color
    cl_dict = {"Reported": "grey80", "Novel": "grey20"}
    clr2 = [cl_dict[x] for x in df["Reported"]]

    layout = df_gr.layout("fr")
    #  https://igraph.org/python/doc/tutorial/tutorial.html
    outname = os.path.join(outf, "combined_network.pdf")
    ig.plot(
        df_gr,
        outname,
        layout=layout,
        vertex_size=probs,
        edge_width=1,
        edge_color=clr2,
        vertex_color=clr,
        vertex_frame_color=clr,
        keep_aspect_ratio=False,
    )


def plot_fdr(tmp_fold, out_fold, target_fdr=0.5):
    """
    plot fdr
    fdr is an array of array
    thresh is an array of array
    path needs to be in output folder/stats
    for each file in path search for fdr.txt file
    """
    # initialize empty figure
    csfont = {"fontname": "sans-serif"}
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    ax.grid(color="grey", linestyle="--", linewidth=0.25, alpha=0.5)
    for sample in [x[0] for x in os.walk(tmp_fold) if x[0] is not tmp_fold]:
        fdrfile = os.path.join(sample, "fdr.txt")
        fdrfile = pd.read_csv(fdrfile, sep="\t", index_col=False)
        ids = os.path.basename(os.path.normpath(sample))
        x_sort, y_sort = zip(*sorted(zip(fdrfile["sumGO"], fdrfile["fdr"])))
        plt.plot(x_sort, y_sort, label=ids, linewidth=1.5, linestyle="-")
    plt.xlabel("GO score", fontsize=14, **csfont)
    plt.ylabel("False Discovery Rate", fontsize=14, **csfont)
    plt.axhline(y=float(target_fdr), linestyle="--", color="black")
    ax.set_xlabel("GO score", fontsize=14, **csfont)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=12)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        prop={"family": "sans-serif", "size": 12},
    )
    fig.savefig(
        os.path.join(out_fold, "FalseDiscoveryRate.pdf"), dpi=800, bbox_inches="tight"
    )
    plt.close()
    return True


def plot_recall(out_fold):
    """
    plot barplot recovery complexes for every condition
    """
    csfont = {"fontname": "sans-serif"}
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    repo = os.path.join(out_fold, "ComplexReport.txt")
    repo = pd.read_csv(repo, sep="\t")
    repo = repo[repo["Is Complex"] == "Positive"]
    sum_e = repo.groupby(['Condition', 'Replicate', 'Reported']).size().to_frame().reset_index()
    sum_e = sum_e.values
    space = 0.3
    conditions = np.unique(sum_e[:, 0])
    repl = np.unique(sum_e[:, 1])
    width = (1 - space) / (len(repl))
    # Create a set of bars at each position
    for i, cond in enumerate(repl):
        indeces = range(1, len(conditions) + 1)
        rep = sum_e[(sum_e[:, 1] == cond) & (sum_e[:, 2] == "Reported")]
        rep = rep[:, 3].astype(np.float)
        nov = sum_e[(sum_e[:, 1] == cond) & (sum_e[:, 2] == "Novel")]
        nov = nov[:, 3].astype(np.float)
        pos = [j - (1 - space) / 2.0 + i * width for j in indeces]
        ax.bar(
            pos,
            rep,
            width=width,
            color="#3C5488B2",
            edgecolor="black",
            linewidth=1,
            label="Reported",
        )
        ax.bar(
            pos,
            nov,
            width=width,
            bottom=rep,
            color="#4DBBD5B2",
            edgecolor="black",
            linewidth=1,
            label="Novel",
        )

    # Set the x-axis tick labels to be equal to the repl
    ax.set_xticks(indeces)
    ax.set_xticklabels(conditions)
    plt.setp(plt.xticks()[1], rotation=0)

    # Add the axis labels
    ax.set_ylabel("# Positive", fontsize=14, **csfont)
    ax.set_xlabel("", fontsize=14, **csfont)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=12)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="upper left")
    fig.savefig(
        os.path.join(out_fold, "RecallDatabase.pdf"), dpi=800, bbox_inches="tight"
    )
    plt.close()
    return True


def runner(tmp_fold, out_fold, target_fdr, sid):
    """
    performs all plots using matplotlib
    plots
    1) combined fdr
    2) recall from db/positive
    3) Profile plots across all replicates
    4) network combined from all conditions
    """
    outf = os.path.join(out_fold, "Plots")
    if not os.path.isdir(outf):
        os.makedirs(outf)
    plot_fdr(tmp_fold, out_fold, target_fdr)
    plot_recall(out_fold)
    comb = os.path.join(tmp_fold, "combined.txt")
    # plot_positive(comb, sid, pl_dir=outf)
    plot_network(out_fold, "PPIreport.txt")
