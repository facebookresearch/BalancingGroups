# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import matplotlib
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.utils.data
from models import ToyNet
from parse import parse_json_to_df
from datasets import Toy
import matplotlib.pyplot as plt
from torch import FloatTensor as FT
import seaborn as sns
from tqdm import tqdm
import itertools


def generate_heatmap_plane(X):
    xlim = np.array([-2, 2])
    ylim = np.array([-2, 2])
    n = 200
    d1, d2 = torch.meshgrid(
        [torch.linspace(xlim[0], xlim[1], n), torch.linspace(ylim[0], ylim[1], n)]
    )
    heatmap_plane = torch.stack((d1.flatten(), d2.flatten()), dim=1)

    # below, we compute the distance of each point to the training datapoints.
    # if the distance is less than 1e-3, that point used the noise dimensions
    # of the closest training point.
    # 10000 x 300
    dists = (heatmap_plane[:, 0:1] - FT(X[:, 0:1].T)) ** 2 + (
        heatmap_plane[:, 1:2] - FT(X[:, 1:2].T)
    ) ** 2
    noise_dims = FT(X)[torch.argmin(dists, 1)][:, 2:] * (
        dists.min(1)[0] < 0.001
    ).unsqueeze(1)
    return torch.cat([heatmap_plane, noise_dims], 1)


def load_model(path):
    state_dict = torch.load(path)
    gammas = [
        state_dict["model"]["network.gammas"].squeeze()[i].item() for i in range(3)
    ]
    model = torch.nn.ModuleDict({"network": ToyNet(1202, gammas)})
    model.load_state_dict(state_dict["model"])
    model = model.network
    model.to(DEVICE)
    return model


def plot(
    exps,
    all_train_envs,
    all_hm,
    gammas,
    heatmap_plane,
    error_df,
    filename="toy_exp",
):
    heatmap = all_hm.mean(1)

    matplotlib.rcParams["contour.negative_linestyle"] = "solid"
    cm = ListedColormap(["#C82506", "#0365C0"])
    plt.rc("font", size=18, family="Times New Roman")
    # plt.figure(figsize=(16, 4.5))
    fig, axs = plt.subplots(2, len(exps), figsize=(4 * len(exps), 8))

    n = int(np.sqrt(heatmap_plane.shape[0]))
    hmp_x = heatmap_plane[:, 0].detach().cpu().numpy().reshape(n, n)
    hmp_y = heatmap_plane[:, 1].detach().cpu().numpy().reshape(n, n)
    hma = heatmap.reshape(-1, n, n).sigmoid()

    for i in range(len(exps)):
        ax = axs[0, i] if len(exps) > 1 else axs[0]
        vmin, vmax = hma[i, -1, -1], hma[i, 1,1]
        delta = vmax-vmin
        vmin, vmax = vmin-0.25*delta, vmax+0.25*delta
        cm = plt.cm.RdBu.copy()
        cm.set_under("#C82506")
        cm.set_over("#0365C0")
        p = ax.contourf(
            hmp_x,
            hmp_y,
            hma[i],
            np.linspace(vmin, vmax, 20),
            cmap=cm,
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
            extend="both"
        )
        ax.contour(
            hmp_x, hmp_y, hma[i], [0.5], antialiased=True, linewidths=1.0, colors="k"
        )
        ax.set_title(exps[i].upper())

        ax.set_xlabel("x spu * gamma spu")
        ax.set_ylabel("x core * gamma core")
        ax.text(-1.7, 1.7, "I", horizontalalignment='center', verticalalignment='center', fontsize=18, color="k")
        ax.text(1.7, 1.7, "II", horizontalalignment='center', verticalalignment='center', fontsize=18, color="k")
        ax.text(-1.7, -1.7, "III", horizontalalignment='center', verticalalignment='center', fontsize=18, color="k")
        ax.text(1.7, -1.7, "IV", horizontalalignment='center', verticalalignment='center', fontsize=18, color="k")
        ax.axhline(y=0, ls="--", lw=0.7, color="k", alpha=0.5)
        ax.axvline(x=0, ls="--", lw=0.7, color="k", alpha=0.5)
        # ax.xaxis.set_major_locator(plt.NullLocator())
        # ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_xlim(np.array([-2, 2]))
        ax.set_ylim(np.array([-2, 2]))
        ticks = [-2, -1, 0, 1, 2]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([int(t * gammas[0]) for t in ticks])
        ax.set_yticklabels([int(t * gammas[1]) for t in ticks])

        for X, y in all_train_envs:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm, edgecolors='none', s=5, alpha=0.3)

        ax_ = axs[1, i] if len(exps) > 1 else axs[1]
        l = sns.lineplot(
            data=error_df.groupby("method").get_group(exps[i]),
            x="epoch",
            y="error",
            hue="phase",
            ax=ax_,
            ci=90
        )
        handles, labels = l.get_legend_handles_labels()
        l.get_legend().remove()
        ax_.grid(color="k", linestyle="--", linewidth=0.5, alpha=0.3)
        ax_.set_title(exps[i].upper())
        # ax_.set_xscale("log")
        ax_.set_xlabel("Iterations")
        ax_.set_ylabel("worst-group-accuracy")
        ax_.set_ylim([-0.005, 1.005])

    lg = fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    
    plt.savefig(f"figures/{filename}.pdf",bbox_extra_artists=(lg,), bbox_inches='tight')
    plt.savefig(f"figures/{filename}.png",bbox_extra_artists=(lg,), bbox_inches='tight')


if __name__ == "__main__":

    seeds = 1
    n_samples = 1000
    dim_noise = 1200
    DEVICE = 0
    gammas = [4, 1.0, 20.0]
    exps = ["erm", "subg", "rwg"]

    df = parse_json_to_df(["toy_sweep"])
    idx = [
        "method",
        "lr",
        "weight_decay",
        "batch_size",
        "init_seed",
        "epoch",
        "file_path",
    ]
    # df.set_index(idx)

    def get_ploting_params(df):
        models = {
            (exp, seed): load_model(path.replace(".pt", ".best.pt"))
            for exp, seed, path in (
                df.groupby(["method", "init_seed", "file_path"]).groups.keys()
            )
        }

        df = (
            df.melt(
                id_vars=idx,
                value_vars=["min_acc_va", "min_acc_te", "min_acc_tr"],
                var_name="phase",
                value_name="error",
            )
            .replace({"min_acc_va": "valid", "min_acc_te": "test", "min_acc_tr": "train"})
            .reset_index()
        )

        datasets = []
        for i in range(seeds):
            torch.manual_seed(i)
            np.random.seed(i)
            d = Toy("tr")
            datasets.append((d.x, d.y))

        all_hm = torch.zeros(len(exps), seeds, 200 * 200)
        for exp_i, exp in enumerate(exps):
            for i in range(seeds):
                heatmap_plane = generate_heatmap_plane(datasets[i][0]).to(DEVICE)
                all_hm[exp_i, i] = models[(exp, i)](heatmap_plane).detach().cpu()
        return exps, datasets, all_hm, gammas, heatmap_plane, df

    groups = df.groupby(
        ["lr", "weight_decay", "batch_size", "gamma_spu", "gamma_core", "gamma_noise"]
    )
    for (lr, wd, bs, gms, gmc, gmn), g_df in groups:
        plot(
            *get_ploting_params(g_df),
            filename=f"toy_sweep_lr_{lr}_wd_{wd}_bs_{bs}_gms_{gms}_gmc_{gmc}_gmn_{gmn}",
        )
