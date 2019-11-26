import os
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib

#  matplotlib.use('Agg')
import pandas as pd
import numpy as np


import PCprophet.io_ as io
import PCprophet.signal_prc as sig


def smart_makefold(path, folder):
    """
    """
    plot_fold = os.path.join(path, folder)
    if not os.path.isdir(plot_fold):
        os.makedirs(plot_fold)
    return plot_fold


def plot_positive(out_fold, tmp_fold, sid):
    """
    loop through all positive in outfile and plots from transf matrix
    """
    repo = os.path.join(out_fold, "ComplexReport.txt")
    comb = os.path.join(tmp_fold, "combined.txt")
    sa_id = pd.read_csv(sid, sep="\t", index_col=False)
    ids = dict(zip(sa_id["short_id"], sa_id["cond"]))
    repo = pd.read_csv(repo, sep="\t", index_col=False)
    comb = pd.read_csv(comb, sep="\t", index_col=False)
    fr2ids = dict(zip(sa_id["cond"], sa_id["fr"]))
    # now we reconvert to original fractions
    comb.drop(["PKS", "SEL", "P", "CMPLT", "GO"], axis=1, inplace=True)
    xx = lambda row: np.array(sig.resize_plot(row["INT"], input_fr=72, output_fr=72))
    comb["INT"] = comb.apply(xx, axis=1)
    if "Treat1" in ids.values():
        positive = set(repo[repo["Is Complex"] == "Positive"]["ComplexID"])
        plot_sec_diff(comb, ids, positive, out_fold)
    for index, row in repo.iterrows():
        newf = row["Condition"] + " " + str(round(row["Replicate"], 1))
        outf = os.path.join(out_fold, newf)
        if not os.path.isdir(outf):
            os.makedirs(outf)
        mask = (
            (comb["CMPLX"] == row["ComplexID"])
            & (comb["COND"] == ids[row["Condition"]])
            & (comb["REPL"] == row["Replicate"])
        )
        filt = comb[mask]
        plot_fold = False
        if row["Reported"] == "Reported" and row["Is Complex"] == "Positive":
            plot_fold = smart_makefold(outf, "Positive reported")
        elif row["Reported"] == "Novel" and row["Is Complex"] == "Positive":
            plot_fold = smart_makefold(outf, "Positive novel")
        elif row["Reported"] == "Reported" and row["Is Complex"] == "Negative":
            plot_fold = smart_makefold(outf, "Negative reported")
            # now we plot the row from comb
        if plot_fold:
            plot_general(
                filt, "INT", plot_fold, row["ComplexID"], row["ComplexID"], False
            )
    return True


def subset_ctrl_fill(df):
    """
    get a df and returns ctrl dataframe filled with all missing 0
    """
    prots = pd.concat([df["ID"], df["COND"], df["INT"]], axis=1)
    prots.drop_duplicates(subset=["ID", "COND"], inplace=True)
    allprot = set(df["ID"])
    ctrl = prots[prots["COND"] == "Ctrl"]
    missing = allprot - set(ctrl["ID"])
    zeroes = [np.zeros(len(ctrl["INT"][0]))] * len(missing)
    #  df.insert(0, 'Group', 'A')
    # series = [pd.Series(mat[name][:, 1]) for name in Variables]
    df = pd.DataFrame({"ID": list(missing), "INT": zeroes})
    df.insert(0, "COND", "Ctrl")
    ctrl = ctrl.append(df, ignore_index=True)
    ctrl.set_index("ID")
    # ctrl.drop('COND')
    return ctrl


def plot_sec_diff(comb, ids, positive, out_fold):
    """
    plot protein level differential subtract each protein to itself and plot
    """
    diff_fold = os.path.join(out_fold, "Differential Plot")
    if not os.path.isdir(diff_fold):
        os.makedirs(diff_fold)
    groups = ["CMPLX", "COND", "ID"]
    mm = (
        comb.groupby(groups, as_index=False)
        .INT.apply(lambda g: np.mean(g.values.tolist(), axis=0))
        .reset_index()
    )
    newids = {v: k for k, v in ids.items()}
    # prot only
    ctrl = subset_ctrl_fill(mm)
    ctrl.set_index("ID", inplace=True)
    for sample in ids.values():
        if sample == "Ctrl":
            continue
        sample_diff = os.path.join(diff_fold, newids[sample])
        if not os.path.isdir(sample_diff):
            os.makedirs(sample_diff)
        filt = mm[mm["COND"] == sample]
        # check here order if up or down when plot
        pairs = [newids[x] for x in ["Ctrl", sample]]
        for cmplx in set(filt["CMPLX"]):
            # now we filt then subtract from prots
            # so now should be only one prot row unique per line
            # now we substract ctrl
            filt_s = filt[(filt["CMPLX"] == cmplx)]
            treat = filt_s.set_index("ID")
            # TODO need to take care of uneven size here....
            treat["INT"] = treat["INT"] - ctrl["INT"]
            treat["ID"] = treat.index
            plot_general(treat, "INT", sample_diff, cmplx, cmplx, pairs, True)


def plot_general(filt, yaxis, fold, ids, nm, pairs, hline=False):
    csfont = {"fontname": "sans-serif"}
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["grid.color"] = "k"
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")
    ax.grid(color="grey", linestyle="--", linewidth=0.25, alpha=0.5)
    fractions = [int(x) for x in range(1, len(filt[yaxis].iloc[0]) + 1)]
    for index, row in filt.iterrows():
        plt.plot(
            fractions, row[yaxis], "-", lw=1, label=str(row["ID"]),
        )
    plt.legend(
        loc="best",
        title="Subunits",
        title_fontsize=16,
        prop={"family": "sans-serif", "size": 16},
    )
    plt.xlabel("Rescaled fraction (arb. unit)", fontsize=14, **csfont)
    title = ax.set_title("\n".join(nm.split("#")), fontsize=18, **csfont)
    if hline:
        ax.set_ylim(-1.0, 1.0)
        plt.ylabel("Delta rescaled intensity", fontsize=14, **csfont)
        ax.axhline(0, color="black")
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([-1, 1])
        ax2.set_yticklabels(pairs, fontsize=14)
    else:
        plt.ylabel("Rescaled intensity", fontsize=14, **csfont)
    if "/" in ids:
        ids = ids.replace("/", " ")
    plotname = os.path.join(str(fold) + "/%s.pdf" % str(ids))
    try:
        fig.savefig(plotname, dpi=600)
    except OSError as exc:
        if exc.errno == 63:
            ids = ids.split("#")[0]
            plotname = os.path.join(str(fold) + "/%s.pdf" % str(ids))
            fig.savefig(plotname, dpi=600)
        else:
            raise
    plt.close()


def plot_alignment(not_ali, ali, peaks, fold, labelrow, plottype):
    csfont = {"fontname": "sans-serif"}
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["grid.color"] = "k"
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 9), facecolor="white", nrows=2)
    not_ali["INT"] = not_ali["no_al"]
    for k in ([ax1, not_ali], [ax2, ali]):
        # select in peaks
        k[0].grid(color="grey", linestyle="--", linewidth=0.25, alpha=0.5)
        fractions = [int(x) for x in range(1, len(k[1]["INT"].iloc[0]) + 1)]
        for index, row in k[1].iterrows():
            tmp_pks = peaks[peaks["COND"] == row[labelrow]]
            y = [row["INT"][int(x)] for x in tmp_pks["SEL"]]
            x_sort, y_sort = zip(*sorted(zip(tmp_pks["SEL"], y)))
            k[0].plot(x_sort, y_sort, "-", label=str(row[labelrow]))
        plt.legend(
            loc="best",
            title="Conditions",
            title_fontsize=16,
            prop={"family": "sans-serif", "size": 16},
        )

    ax1.set_ylabel("Pre alignement", fontsize=14, **csfont)
    ax2.set_ylabel("Post alignement", fontsize=14, **csfont)
    ax2.set_xlabel("Fraction (arb. unit)", fontsize=14, **csfont)
    ax1.yaxis.set_label_coords(-0.1, 0.5)
    ax2.yaxis.set_label_coords(-0.1, 0.5)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plotname = os.path.join(fold, "%sAlignement.pdf" % str(plottype))
    fig.savefig(plotname, dpi=600)
    plt.close()


# @io.timeit
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
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")
    ax.grid(color="grey", linestyle="--", linewidth=0.25, alpha=0.5)
    for sample in [x[0] for x in os.walk(tmp_fold) if x[0] is not tmp_fold]:
        fdrfile = os.path.join(sample, "fdr.txt")
        fdrfile = pd.read_csv(fdrfile, sep="\t", index_col=False)
        ids = os.path.basename(os.path.normpath(sample))
        x_sort, y_sort = zip(*sorted(zip(fdrfile["sumGO"], fdrfile["fdr"])))
        plt.plot(x_sort, y_sort, label=ids, linewidth=1.5, linestyle="-")
    plt.legend(loc="best", prop={"family": "sans-serif", "size": 16})
    plt.xlabel("GO score", fontsize=14, **csfont)
    plt.ylabel("False Discovery Rate positive", fontsize=14, **csfont)
    plt.axhline(y=float(target_fdr), linestyle="--", color="black")
    fig.savefig(os.path.join(out_fold, "FalseDiscoveryRate.pdf"), dpi=600)
    plt.close()
    return True


def mean_subs(g):
    return np.mean(g.values.tolist(), axis=0)


def conv2float(row):
    row["INT"] = [float(x) for x in row["INT"]]
    row["no_al"] = [float(x) for x in row["no_al"]]
    return row


def chained_mean_groupby(df, what, group1, group2):
    """
    calculate mean once by groupby with group1 then with group2
    """
    df = df.groupby(group1, as_index=True)[what].apply(mean_subs).reset_index()
    df = df.groupby(group2, as_index=True)[what].apply(mean_subs).reset_index()
    return df


def plot_recall(out_fold):
    """
    plot barplot recovery complexes
    """
    csfont = {"fontname": "sans-serif"}
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    repo = os.path.join(out_fold, "ComplexReport.txt")
    repo = pd.read_csv(repo, sep="\t", index_col=False)
    repo = repo[repo["Is Complex"] == "Positive"]
    sum_e = repo.groupby(["Condition", "Replicate", "Reported"]).size().reset_index()
    sum_e = sum_e.values
    space = 0.3
    conditions = np.unique(sum_e[:, 0])
    repl = np.unique(sum_e[:, 1])
    n = len(repl)
    width = (1 - space) / (len(repl))
    # Create a set of bars at each position
    for i, cond in enumerate(repl):
        indeces = range(1, len(conditions) + 1)
        rep = sum_e[(sum_e[:, 1] == cond) & (sum_e[:,2] == 'Reported')]
        rep = rep[:, 3].astype(np.float)
        nov = sum_e[(sum_e[:, 1] == cond) & (sum_e[:,2] == 'Novel')]
        nov = nov[:, 3].astype(np.float)
        pos = [j - (1 - space) / 2.0 + i * width for j in indeces]
        ax.bar(pos, rep, width=width, color="#ff7f0e", edgecolor="black", linewidth=1, label='Reported')
        ax.bar(pos, nov, width=width, bottom=rep, color="#1f77b4", edgecolor="black", linewidth=1, label='Novel')
    # Set the x-axis tick labels to be equal to the repl
    ax.set_xticks(indeces)
    ax.set_xticklabels(conditions)
    plt.setp(plt.xticks()[1], rotation=0)

    # Add the axis labels
    ax.set_ylabel("# Positive", fontsize=14, **csfont)
    ax.set_xlabel("", fontsize=14, **csfont)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left")
    fig.savefig(os.path.join(out_fold, "RecallDatabase.pdf"), dpi=800)
    plt.show()
    plt.close()
    return True

def plot_recalibration(tmp_fold, out_fold):
    """
    plot average profile of complexes used in aligner
    """
    toread = os.path.join(tmp_fold, "complex_align.txt")
    try:
        common = set(pd.read_csv(toread, sep="\t", index_col=False)["CMPLX"])
    except FileNotFoundError as e:
        return True
    toread = os.path.join(tmp_fold, "combined.txt")
    comb = pd.read_csv(toread, sep="\t", index_col=False)
    comb = comb[comb["CMPLX"].isin(common)]
    comb["INT"] = comb["INT"].str.split("#", expand=False)
    comb["no_al"] = comb["no_al"].str.split("#", expand=False)
    comb = comb.apply(conv2float, axis=1)
    # this basically takes care of multiple replicates
    groups = ["CMPLX", "COND", "REPL", "ID"]
    # replicates plot
    aligned = chained_mean_groupby(comb, "INT", groups, ["COND", "REPL"])
    not_aligned = chained_mean_groupby(comb, "no_al", groups, ["COND", "REPL"])
    # now we add the peaks
    peaks = chained_mean_groupby(comb, "SEL", groups, ["COND", "CMPLX"])
    peaks["SEL"] = peaks.SEL.round()
    plot_alignment(not_aligned, aligned, peaks, out_fold, "COND", "Condition")


def runner(tmp_fold, out_fold, target_fdr, sid):
    """
    performs all plots using matplotlib
    plots
    1) combined fdr
    2) recall from db/positive
    3) Positive Reported
    ø4) Possitive Novel
    5) negative Reported
    6) Apex plot => positive negative across the entire sec
    """
    plot_fdr(tmp_fold, out_fold, target_fdr)
    plot_recall(out_fold)
    plot_recalibration(tmp_fold, out_fold)
    assert False
    # now fails
    plot_positive(out_fold, tmp_fold, sid)
    #  plot_differential(out_fold, tmp_fold, sid)
