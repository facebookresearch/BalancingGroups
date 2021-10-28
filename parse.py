# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#!/usr/bin/env python

import os
import glob
import json
import argparse
from typing import ContextManager
import pandas as pd
from pandas.core.indexes import multi
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import alexandergovern
from matplotlib import cm
from pandas.api.types import is_numeric_dtype


def remove(lis, val):
    return [value for value in lis if value != val]


def anova_test(best_df, df, metric):
    for (dataset, groups), dataset_df in best_df.groupby(level=["dataset", "Groups"]):
        metric_values = [
            df.get_group(idx)[metric].values for idx, _ in dataset_df.iterrows()
        ]
        best_df.loc[(dataset, slice(None), slice(None), groups), "Signif Diff"] = (
            alexandergovern(*metric_values).pvalue < 0.05
        )
    return best_df


def convert_df_to_readable_format(reduced, bold=None, latex=None):
    # Formatting table contents with mean (std)
    summary = pd.DataFrame()
    pm_sign = "$\\pm$" if latex else "+/-"

    for c in reduced.columns.get_level_values(0):
        if "mean" in reduced[c] and "std" in reduced[c]:
            if "acc" in c.lower():
                summary[c] = (
                    (100 * reduced[c]["mean"]).map("{:.1f}".format)
                    + pm_sign
                    + (100 * reduced[c]["std"]).map("{:.1f}".format)
                )
            else:
                summary[c] = (
                    reduced[c]["mean"].map("{:.1f}".format)
                    + pm_sign
                    + reduced[c]["std"].map("{:.1f}".format)
                )
        elif "min" in reduced[c]:
            summary[c + " range"] = (
                "["
                + reduced[c]["min"].map("{:.1f}".format)
                + ", "
                + reduced[c]["max"].map("{:.1f}".format)
                + "]"
            )
        else:
            if is_numeric_dtype(reduced[c]) and reduced[c].dtype == "float":
                summary[c] = reduced[c].map("{:.1f}".format)
            else:
                summary[c] = reduced[c]
    if bold:
        if latex:
            bold_l, bold_r = r"\textbf{", "}"
        else:
            bold_l, bold_r = "*", ""

        best_algos = (
            reduced.sort_values((bold["best_metric"], "mean"), ascending=bold["order"])
            .groupby(bold["best_metric_group"])
            .head(1)
            .index
        )
        summary.loc[best_algos, bold["best_metric"]] = summary.loc[
            best_algos, bold["best_metric"]
        ].map(lambda x: bold_l + x + bold_r)
    return summary


def final_latex_table(final_df, df, do_anova, col_to_show):
    template_begining = (
        r"""
    \begin{tabular}{lllccccc}
                \toprule
                \textbf{Method} & \textbf{\#HP} & \textbf{Groups} & \multicolumn{4}{c}{\textbf{Worst Acc}} & \textbf{Average}                                                                                                          \\
                \cmidrule(lr){4-7}
                                &               &                 & CelebA                                 & Waterbirds       & MultiNLI                                       & CivilComments                                  &      \\
                \midrule
    """
    )
    middle = r""
    last_group = None
    df = df.set_index(["dataset", "Method"])
    for _, row in final_df.iterrows():
        for dataset in ["CelebA", "Waterbirds", "MultiNLI", "CivilComments"]:
            if do_anova:
                if df.loc[(dataset, row["Method"])]["Signif Diff"].item():
                    row[dataset] = "\cellcolor{blue!7}" + str(
                        row[dataset]
                    )
        if row["Groups"] != last_group and last_group is not None:
            middle += "\\midrule \n"
        middle += r" & ".join(row.astype(str).values)
        middle += "\\\\ \n"
        last_group = row["Groups"]

    template_ending = r"""
    \bottomrule \\
    \end{tabular}
    """
    return template_begining + middle + template_ending


def parse_json_to_df(dirs):
    records = []
    groups = {
        "erm": "No",
        "jtt": "No",
        "suby": "No",
        "rwy": "No",
        "dro": "Yes",
        "rwg": "Yes",
        "subg": "Yes",
    }
    nb_hps = {"erm": 4, "jtt": 6, "suby": 4, "rwy": 4, "dro": 5, "rwg": 4, "subg": 4}

    for dname in dirs:
        for fname in glob.glob(os.path.join(dname, "*.out")):
            with open(fname, "r") as f:
                lines = f.readlines()

            for line in lines:
                if not line.startswith("{"):
                    continue

                record = json.loads(line)
                this_row = dict(record["args"])
                this_row["epoch"] = record["epoch"]
                this_row["time"] = record["time"] / 3600
                this_row["min_acc_va"] = min(record["acc_va"])
                this_row["min_acc_tr"] = min(record["acc_tr"])
                this_row["avg_acc_va"] = record["avg_acc_va"]
                this_row["min_acc_te"] = min(record["acc_te"])
                this_row["avg_acc_te"] = record["avg_acc_te"]
                this_row["Groups"] = groups[this_row["method"]]
                this_row["#HP"] = nb_hps[this_row["method"]]
                this_row["file_path"] = os.path.splitext(fname)[0] + ".pt"
                records.append(this_row)
    if not len(records):
        quit()

    pd.set_option(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    )

    return pd.DataFrame(records)


def reorganize_df(df, col_to_show=None):
    df = (
        df.set_index(["dataset", "Method", "#HP", "Groups"])[col_to_show]
        .unstack(level=0)
        .sort_index(axis=0, level=2)
    )
    df.columns = df.columns.set_names(None)
    df = df.sort_index(axis=1)
    # df = df.reindex(['Worst Acc', 'Time (h)', 'Signif Diff'], level=1, axis=1)
    df = df.reindex(["CelebA", "Waterbirds", "MultiNLI", "CivilComments"], axis=1)

    df = df.reset_index()
    return df


def model_paths(df, run_groups):
    models_to_save = []
    for idx, row in df.iterrows():
        models_to_save.append(run_groups.get_group(idx)["file_path"])
    return pd.concat(models_to_save)


def print_hp_table(df, aggregate=True):
    hps = ["lr", "weight_decay", "epoch", "batch_size"]
    hparams = df[[(hp, "mean") for hp in hps]].droplevel(1, axis=1)

    hparams = hparams.apply(
        {
            "lr": np.log10,
            "weight_decay": np.log10,
            "epoch": lambda x: x,
            "batch_size": lambda x: x,
        }
    )
    if aggregate:
        hparams = hparams.groupby(["dataset", "Groups", "method"]).agg(["mean", "std"])

        metric = ("min_acc_te", "mean")
        hparams[("min_acc_te", "min")] = (
            df.groupby(["dataset", "Groups", "method"]).min()[metric] * 100
        )
        hparams[("min_acc_te", "max")] = (
            df.groupby(["dataset", "Groups", "method"]).max()[metric] * 100
        )
        hparams[("min_acc_te_delta", "")] = (
            hparams[("min_acc_te", "max")] - hparams[("min_acc_te", "min")]
        )
    else:
        hparams = pd.concat([hparams, df[["min_acc_te"]]], axis=1)
        hparams.columns = pd.MultiIndex.from_tuples(
            [(hp, "") for hp in hps] + df[["min_acc_te"]].columns.tolist()
        )
        hparams = hparams.droplevel(["hparams_seed", "#HP"], axis=0)
        hparams = hparams.reorder_levels(["dataset", "Groups", "method"])
        # print(hparams)
    hparams = hparams.sort_index()
    print(convert_df_to_readable_format(hparams))
    df = convert_df_to_readable_format(hparams, latex=True)

    cmaps = {
        "lr": "bone",
        "weight_decay": "pink",
        "epoch": "bone",
        "batch_size": "pink",
    }
    groups = hparams.groupby(["dataset"])

    for idx, row in hparams.iterrows():
        for hp in ["lr", "weight_decay", "batch_size", "epoch"]:
            cmap = cm.get_cmap(cmaps[hp])
            hp_tup = (hp, "mean") if aggregate else hp
            scale = {
                "min": groups.get_group(idx[0])[hp_tup].min().item(),
                "max": groups.get_group(idx[0])[hp_tup].max().item(),
            }

            max_level = {
                "lr": 1 / 6,
                "weight_decay": 1 / 6,
                "batch_size": 1 / 6,
                "epoch": 1 / 6,
            }[hp]
            if hp in ["weight_decay", "batch_size"]:
                level = 1 - (
                    max_level
                    * (row[hp_tup].item() - scale["min"])
                    / (scale["max"] - scale["min"])
                )
            else:
                level = 1 + (
                    max_level
                    * (row[hp_tup].item() - scale["max"])
                    / (scale["max"] - scale["min"])
                )
            color = ["{:.3f}".format(c) for c in cmap(level)[:3]]
            df.loc[idx, hp] = (
                "\cellcolor[rgb]{" + ",".join(color) + "}" + str(df.loc[idx, hp])
            )
    filename = "hp_table_mean" if aggregate else "hp_table"
    df.to_latex(f"tables/{filename}.tex", multicolumn=True, multirow=True, escape=False)


def plot_min_acc_evol(best_df, all_runs, filename):
    df = []
    all_runs_groups = all_runs.groupby(best_df.index.names)

    for idx, _ in best_df.iterrows():
        df.append(all_runs_groups.get_group(idx))
    df = (
        pd.concat(df)
        .sort_index()
        .reindex(["CelebA", "Waterbirds", "MultiNLI", "CivilComments"], level="dataset")
    )

    groups = df.groupby(
        ["dataset", "method", "hparams_seed", "init_seed", "Groups", "#HP"]
    )
    windows = {
        "CelebA": 5,
        "Waterbirds": 10,
    }
    dfs = []
    for group, df_group in groups:
        if group[0] in windows:
            dfs.append(df_group.rolling(window=windows[group[0]]).mean())
        else:
            dfs.append(df_group)
    df = pd.concat(dfs)
    plt.rc("font", size=11)
    df = (
        df.melt(
            value_vars=["min_acc_te", "min_acc_tr"],
            var_name="phase",
            value_name="worst-group-acc",
            ignore_index=False,
        )
        .replace({"min_acc_te": "test", "min_acc_tr": "train"})
        .reset_index()
    )

    sns.set_theme(context="talk", style="white", font="Times New Roman")

    scale = 1
    # plt.figure(figsize=(scale * 8, scale * 11))

    g = sns.relplot(
        data=df,
        x="epoch",
        y="worst-group-acc",
        hue="method",
        style="phase",
        kind="line",
        row="Groups",
        col="dataset",
        height=scale * 3.5,
        aspect=1,
        facet_kws=dict(sharex=False, sharey=False, margin_titles=True),
        alpha=0.7,
    )
    g.set_axis_labels("epoch", "worst-group-acc")
    g.set_titles(row_template="Groups = {row_name}", col_template="{col_name}")
    # g.add_legend(loc="lower center", ncol=4)
    g.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", dpi=300)
    plt.savefig(f"figures/{filename}.png", dpi=300)


def format_result_tables(df, run_groups, do_anova=False):
    if do_anova:
        df = anova_test(df, run_groups, "min_acc_te")
    df = df.reset_index()

    if not args.full:
        df = df[["dataset", "method", "Groups", "#HP", "min_acc_te", "time"]]

    df = df.rename(
        columns={"min_acc_te": "Worst Acc", "time": "Time (h)", "method": "Method"}
    )
    multip = 100 if args.col_to_show == "Worst Acc" else 1
    avg_accs_per_method = (
        (multip * df.groupby("Method").mean()[(args.col_to_show, "mean")])
        .map("{:.1f}".format)
        .reset_index(name="Average")
    )
    if args.bold:
        bold = {
            "best_metric": args.col_to_show,
            "order": False if "acc" in args.col_to_show.lower() else True,
            "best_metric_group": ["dataset", "Groups"],
        }
    else:
        bold = False
    term_df = convert_df_to_readable_format(df, bold, latex=False)
    term_df = reorganize_df(term_df, col_to_show=args.col_to_show)
    term_df = term_df.merge(avg_accs_per_method, on="Method", how="left")
    print(term_df)

    latex_df = convert_df_to_readable_format(df, bold, latex=True)
    latex_df = reorganize_df(latex_df, col_to_show=args.col_to_show)
    latex_df = latex_df.merge(avg_accs_per_method, on="Method", how="left")

    os.makedirs("tables", exist_ok=True)
    open(
        f'tables/result_{args.col_to_show.replace(" ", "_").replace("(","").replace(")","").lower()}_1.tex',
        "w",
    ).write(final_latex_table(latex_df, df, do_anova, args.col_to_show))


def format_time_results(df_all_epochs, unique_run_id):
    time_delta = df_all_epochs.groupby(unique_run_id)["time"].diff() * 60
    time_delta = time_delta[
        time_delta > 0
    ]  # Remove negative values coming from preemption
    total_time = time_delta.sum().item()
    print("Total compute time : " + str(total_time))
    time_result = time_delta.groupby(["dataset", "method", "#HP", "Groups"]).median()
    average = (
        time_result.groupby(["method", "#HP", "Groups"]).mean().to_frame("Average")
    )
    time_result = time_result.unstack("dataset").sort_index(level="Groups")
    time_result = time_result.join(average).apply(lambda x: x.map("{:.2f}".format))
    print(time_result)
    time_result.to_latex(
        "tables/result_time_h.tex", escape=False, multirow=True, multicolumn=True
    )

    sns.set(style="whitegrid", context="talk")
    g = sns.catplot(
        data=time_delta.to_frame("time").reset_index(),
        x="method",
        y="time",
        col="dataset",
        kind="box",
        sharex=True,
        sharey=False,
        height=6,
        col_wrap=2,
    )
    for ax in g.fig.axes:
        ax.set_yscale("log")
        ax.tick_params(axis="x", labelrotation=45)
    g.set_axis_labels("Method", "Time per epoch in minutes")
    g.set_titles(col_template="{col_name}")
    g.tight_layout()
    plt.savefig(f"figures/time_per_epoch.pdf", dpi=300)
    plt.savefig(f"figures/time_per_epoch.png", dpi=300)


def plot_min_acc_dist(df, run_groups, n):
    dfs = []
    for idx, _ in df.iterrows():
        dfs.append(run_groups.get_group(idx)["min_acc_te"])
    df = pd.concat(dfs).sort_index(level="Groups")
    df = df.reindex(
        ["CelebA", "Waterbirds", "MultiNLI", "CivilComments"], level="dataset"
    ).reset_index()
    sns.set(style="whitegrid", context="talk", font="Times New Roman")
    g = sns.catplot(
        data=df,
        x="method",
        y="min_acc_te",
        col="dataset",
        kind="box",
        sharex=True,
        sharey=False,
        height=4.5,
    )
    for ax in g.fig.axes:
        ax.tick_params(axis="x", labelrotation=45)
    g.set_axis_labels("Method", "worst-group-acc")
    g.set_titles(col_template="{col_name}")
    g.tight_layout()
    plt.savefig(f"figures/worst_group_acc_dist_dataset_{n}.pdf", dpi=300)
    plt.savefig(f"figures/worst_group_acc_dist_dataset_{n}.png", dpi=300)

    plt.figure()
    g = sns.catplot(data=df, x="method", y="min_acc_te", kind="box", height=5.5)
    for ax in g.fig.axes:
        ax.tick_params(axis="x", labelrotation=45)
    g.set_axis_labels("Method", "worst-group-acc")
    g.tight_layout()
    plt.savefig(f"figures/worst_group_acc_dist_{n}.pdf", dpi=300)
    plt.savefig(f"figures/worst_group_acc_dist_{n}.png", dpi=300)


def print_unfinished_runs(dir):
    errored_runs = []
    for d in dir:
        l = os.popen(f"grep -il error {d}/*.err").read()
        l = [o for o in l.split("\n") if o]
        errored_runs.extend(l)
    # unfinished_runs = []
    for run in errored_runs:
        run_json = os.path.splitext(run)[0] + ".out"
        with open(run_json) as f:
            last_epoch = f.readlines()[-1]
        last_epoch = json.loads(last_epoch)
        if last_epoch["epoch"] + 1 != last_epoch["args"]["num_epochs"]:
            print(run_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse sweep folder")
    parser.add_argument("dir", type=str, nargs="+")
    parser.add_argument("--selector", type=str, default="min_acc_va")
    parser.add_argument("--metric", type=str, default="min_acc_te")
    parser.add_argument("--col_to_show", type=str, default="Worst Acc")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--last_epoch", action="store_true")
    parser.add_argument("--do_anova", action="store_true")
    parser.add_argument("--bold", action="store_true")
    parser.add_argument("--small_weight_decay", action="store_true")
    parser.add_argument("--small_lr", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "format_results",
            "format_time_results",
            "best_model_paths",
            "best_mean_model_paths",
            "print_hp_table",
            "unfinished_runs",
            "plot_min_acc_evol",
            "plot_min_acc_dist",
        ],
        default="format_results",
    )
    args = parser.parse_args()

    if args.mode == "unfinished_runs":
        print_unfinished_runs(args.dir)
        exit()

    df = parse_json_to_df(args.dir)
    if args.small_weight_decay:
        df = df[df["weight_decay"] == 1e-4]
    if args.small_lr:
        df = df[df["lr"] == 1e-5]

    unique_run_id = ["dataset", "method", "hparams_seed", "init_seed", "Groups", "#HP"]

    # Renaming datasets
    df = df.replace(
        {
            "celeba": "CelebA",
            "waterbirds": "Waterbirds",
            "multinli": "MultiNLI",
            "civilcomments": "CivilComments",
        }
    )

    df["method"] = df["method"].str.upper()
    df = df.replace({"DRO": "gDRO"})

    df_all_epochs = df.set_index(unique_run_id + ["epoch"])

    df = (
        df.sort_values(by="epoch")
        if args.last_epoch
        else df.sort_values(by=args.selector)
    )
    df = df.groupby(unique_run_id).tail(1).set_index(unique_run_id)

    df_all = df

    # Averaging over init seeds
    run_groups = df.groupby(remove(unique_run_id, "init_seed"))
    df = run_groups.agg(["mean", "std"])
    # Selecting best hyperparmeters in average
    df = df.sort_values(by=["dataset", "method", (args.selector, "mean")])
    df = df.groupby(["dataset", "method"]).tail(args.n)

    if args.mode == "best_model_paths":
        best_models = (
            df_all.sort_values(by=["dataset", "method", args.selector])
            .groupby(["dataset", "method"])
            .tail(args.n)
        )
        # print(best_models)
        for path in best_models["file_path"].values:
            print(path)
    elif args.mode == "best_mean_model_paths":
        best_model_paths = model_paths(df, run_groups)
        for path in best_model_paths.values:
            print(path)
    elif args.mode == "print_hp_table":
        print_hp_table(df, aggregate=(args.n > 1))
    elif args.mode == "format_results":
        format_result_tables(df, run_groups, args.do_anova)
    elif args.mode == "format_time_results":
        format_time_results(df_all_epochs, unique_run_id)
    elif args.mode == "plot_min_acc_evol":
        plot_min_acc_evol(
            df,
            df_all_epochs,
            "worst_acc_evol" if args.n == 1 else f"worst_acc_evol_mean{args.n}",
        )
    elif args.mode == "plot_min_acc_dist":
        plot_min_acc_dist(df, run_groups, args.n)
