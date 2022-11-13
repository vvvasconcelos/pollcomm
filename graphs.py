#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 28/05/2022
# ---------------------------------------------------------------------------
""" graphs.py

Script to generate some graphs for the thesis
"""
# ---------------------------------------------------------------------------
import copy
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import matplotlib.cm as cm
import numpy as np
import util

import experiments as exp
import pollcomm as pc

# source of cyclers and plt parameters: https://ranocha.de/blog/colors/
from cycler import cycler

# "#E69F00": yellow-orange
# "#56B4E9": light blue
# "#009E73": green
# "#0072B2": dark blue
# "#D55E00": red-orange
# "#CC79A7": pink
# "#F0E442": yellow
line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))
marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))

plt.rc("axes", prop_cycle=line_cycler)
# plt.rc("axes", prop_cycle=marker_cycler)
plt.rc("font", family="serif", size=12.)
plt.rc("savefig", dpi=200)
plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
plt.rc("lines", linewidth=2, markersize=10, markeredgewidth=2.5)
plt.rc("axes.spines", top=False, right=False)
# mpl.rcParams['axes.spines.top'] = False



def holling(save_fig=False):

    x = np.linspace(0, 5, 100)
    h_max = 0.2
    h = x / (1 + (1/h_max)*x)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.set_prop_cycle(line_cycler)
    ax.plot(x, h)
    # ax.plot(x+0.1, h)
    # ax.plot(x+0.2, h)
    # ax.plot(x+0.3, h)
    # ax.plot(x+0.4, h)
    # ax.plot(x+0.5, h)
    # ax.plot(x+0.6, h)
    # ax.plot(x+0.7, h)
    ax.hlines(h_max, x.min(), x.max(), colors="black", linestyles="dashed")
    ax.text(
        x.max()-1.5, h_max+0.006, f"$h_{{\mathrm{{max}}}}={h_max}$", color="black"
    )
    ax.set_ylim(0, h_max+0.025)
    ax.set_xlim(0, x.max())
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$H(\rho)$")

    if save_fig:
        plt.savefig("figures/holling.pdf", format="pdf")


def adjacency_matrix(save_fig=False):

    rng = np.random.default_rng()
    N = 0.6
    FL = 0.3
    network, network_FL = pc.nested_network(25, 25, 0.15, FL, N, rng)

    # sort on generalists
    network, network_FL = pc.sort_network(network, network_FL)

    fig, (ax1, ax2) = plt.subplots(
        ncols=2, figsize=(8, 6), constrained_layout=True
    )
    ax1.set_title(f"Nestedness $={N}$", size=16)
    ax1.matshow(network, cmap="Greys", origin="lower")
    ax1.set_xlabel("Pollinator species")
    ax1.set_ylabel("Plant species")
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xticks([4, 9, 14, 19, 24], [5, 10, 15, 20, 25])
    ax1.set_yticks([4, 9, 14, 19, 24], [5, 10, 15, 20, 25])

    ax2.set_title(f"Forbidden Links $={FL}$", size=16)
    ax2.matshow(network_FL, cmap="Greys", origin="lower")
    ax2.set_xlabel("Pollinator species")
    ax2.set_ylabel("Plant species")
    ax2.xaxis.set_ticks_position('bottom')
    ax2.set_xticks([4, 9, 14, 19, 24], [5, 10, 15, 20, 25])
    ax2.set_yticks([4, 9, 14, 19, 24], [5, 10, 15, 20, 25])

    if save_fig:
        plt.savefig("figures/adjacency_matrix.png", format="png")


def beta_plot(save_fig=False, format="png"):

    seed = np.random.SeedSequence().generate_state(1)[0]
    seed = 1174360371
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 15
    N_a = 35
    mu = 0.0001
    connectance = 0.15
    nestedness = 0.6
    nu = 0.8
    q = 0
    AM = pc.AdaptiveModel(
        N_p, N_a, mu, 0.15, 0.3, 0.6, network_type="nested", rng=rng,
        seed=seed, nu=nu, G=1, q=q, feasible=True
    )

    alpha_init = AM.alpha
    beta_P, beta_A = AM.beta_P, AM.beta_A

    # initial conditions
    A_init = 0.1
    rate = 0.05

    # drivers of decline
    dA = {
        "func": lambda t, r: r*t,
        "args": (rate,)
    }

    y0 = np.full(AM.N, A_init, dtype=float)
    y0 = np.concatenate((y0, copy.deepcopy(AM.alpha.flatten())))
    sol_0 = AM.solve(1000, dA=dA, y0=y0, save_period=0, stop_on_collapse=True)
    alpha_end_0 = sol_0.y_partial[:, -1].reshape((AM.N_p, AM.N_a))

    # initial conditions
    A_init = 0.1
    rate = 0.3

    # drivers of decline
    dA = {
        "func": lambda t, r: r*t,
        "args": (rate,)
    }
    sol_1 = AM.solve(1000, dA=dA, y0=y0, save_period=0, stop_on_collapse=True)
    alpha_end_1 = sol_1.y_partial[:, -1].reshape((AM.N_p, AM.N_a))

    fig, ax1 = plt.subplots(
        figsize=(7, 5), constrained_layout=True
    )
    for i in range(AM.N_p, AM.N):
        line_polls, = ax1.plot(sol_0.t, sol_0.y[i], linewidth=1.5)

    ax1.set_xlabel("Time [-]")
    ax1.set_ylabel("Pollinator abundance")
    ax1.set_xlim(sol_0.t.min(), sol_0.t.max())
    ax1.set_ylim(0, 2.5)
    # ax2.set_xticks([15, 30, 45, 60, 75])
    # ax2.set_yticks([0, 0.5, 1, 1.5, 2.0, 2.5])
    if save_fig:
        plt.savefig(f"figures/poll_0.{format}", format=f"{format}")

    fig, ax1 = plt.subplots(
        figsize=(7, 5), constrained_layout=True
    )
    for i in range(AM.N_p, AM.N):
        line_polls, = ax1.plot(sol_1.t, sol_1.y[i], linewidth=1.5)

    ax1.set_xlabel("Time [-]")
    ax1.set_ylabel("Pollinator abundance")
    ax1.set_xlim(sol_1.t.min(), sol_1.t.max())
    ax1.set_ylim(0, 2.5)
    # ax2.set_xticks([15, 30, 45, 60, 75])
    # ax2.set_yticks([0, 0.5, 1, 1.5, 2.0, 2.5])
    if save_fig:
        plt.savefig(f"figures/poll_1.{format}", format=f"{format}")

    alpha_sort = copy.deepcopy(alpha_init)
    alpha_sort[alpha_sort < 0.01] = 0
    alpha_sort[alpha_sort >= 0.01] = 1
    alpha_sort, alpha_init, alpha_end_0, alpha_end_1 = pc.sort_network(
        alpha_sort, alpha_init, alpha_end_0, alpha_end_1
    )

    cmap = "viridis"
    vmin = min((alpha_end_0-alpha_init).min(), (alpha_end_1-alpha_init).min())
    vmax = max((alpha_end_0-alpha_init).max(), (alpha_end_1-alpha_init).max())
    print(vmin, vmax)
    # normalizer = Normalize(
    #     min(alpha_end_0.min(), alpha_end_1.min()),
    #     max(alpha_end_0.max(), alpha_end_1.max())
    # )
    # im_map = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    fig, ax1 = plt.subplots(
        figsize=(7, 5), constrained_layout=True
    )
    im1 = ax1.matshow(
        alpha_end_0-alpha_init, cmap=cmap, origin="lower",
        vmin=vmin, vmax=vmax
    )
    # ax1.set_title(r"$\alpha$ init")
    ax1.set_xlabel("Pollinators")
    ax1.set_ylabel("Plants")
    ax1.set_xticks(
        [4, 9, 14, 19, 24, 29, 34],
        ["5", "10", "15", "20", "25", "30", "35"]
    )
    ax1.set_yticks(
        [4, 9, 14],
        ["5", "10", "15"]
    )
    ax1.xaxis.set_ticks_position('bottom')
    fig.colorbar(im1, ax=ax1,fraction=0.025, pad=0.04)
    # plt.colorbar(im1, ax=ax1, fraction=0.05)
    if save_fig:
        plt.savefig(f"figures/alpha_end_0.{format}", format=f"{format}")

    fig, ax1 = plt.subplots(
        figsize=(7, 5), constrained_layout=True
    )
    im1 = ax1.matshow(
        alpha_end_1-alpha_init, cmap=cmap, origin="lower",
        vmin=vmin, vmax=vmax
    )
    # ax1.set_title(r"$\alpha$ init")
    ax1.set_xlabel("Pollinators")
    ax1.set_ylabel("Plants")
    ax1.set_xticks(
        [4, 9, 14, 19, 24, 29, 34],
        ["5", "10", "15", "20", "25", "30", "35"]
    )
    ax1.set_yticks(
        [4, 9, 14],
        ["5", "10", "15"]
    )
    ax1.xaxis.set_ticks_position('bottom')
    fig.colorbar(im1, ax=ax1,fraction=0.025, pad=0.04)
    if save_fig:
        plt.savefig(f"figures/alpha_end_1.{format}", format=f"{format}")

    # im2 = ax2.matshow(alpha_end_0-alpha_init, cmap=cmap, origin="lower")
    # ax2.set_title(r"$\alpha$ end 0")
    # # ax2.set_ylabel("Plants")
    # ax2.xaxis.set_ticks_position('bottom')
    # # ax2.set_xlabel("Pollinators")
    # plt.colorbar(im2, ax=ax2, fraction=0.05)
    #
    # im3 = ax3.matshow(alpha_end_1-alpha_init, cmap=cmap, origin="lower")
    # ax3.set_title(r"$\alpha$ end 1")
    # ax3.set_ylabel("Plants")
    # ax3.xaxis.set_ticks_position('bottom')
    # ax3.set_xlabel("Pollinators")
    # plt.colorbar(im3, ax=ax3, fraction=0.05)

    # im4 = ax4.matshow(alpha * beta_A, cmap=cmap, origin="lower")
    # ax4.set_title(r"$\alpha \beta^A$")
    # ax4.xaxis.set_ticks_position('bottom')
    # ax4.set_xlabel("Pollinators")
    # plt.colorbar(im4, ax=ax4, fraction=0.05)
    #
    # if save_fig:
    #     plt.savefig(f"figures/betas.png", format="png")


def plot_abundance_rate_critical_dA(recalculate=False, save_fig=False, format="png"):

    seed = np.random.SeedSequence().generate_state(1)[0]
    seed = 3892699245
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 25
    N_a = 25
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    G = 1
    nu = 0.9

    qs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rates = np.linspace(0.0001, 0.5, 81)
    A_init = np.linspace(0, 1, 41)
    if recalculate:

        AM = pc.AdaptiveModel(
            N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng, G=G,
            nu=nu, q=0, feasible=True, feasible_iters=50
        )

        for i, q in enumerate(qs):
            print(f"\nCalculating q: {i+1} out of {len(qs)}...")

            # set q parameter, set seed and rng
            AM.q = q
            AM.rng = rng
            AM.seed = seed

            fname = f"output/abundance_rate_critical_dA_G{AM.G}_nu{AM.nu}_q{AM.q}"
            exp.state_space_abundance_rate_critical_dA(
                fname, AM, rates=rates, A_init=A_init
            )

    dAs_criticals = []
    for i, q in enumerate(qs):
        fname = f"output/abundance_rate_critical_dA_G{G}_nu{nu}_q{q}"
        try:
            with np.load(fname + ".npz") as sol:
                rates = sol["rates"]
                A_init = sol["A_init"]
                dAs_critical = sol["dAs_critical"]
        except FileNotFoundError:
            print(
                f"File for q = {q} not found. Make sure there exists an output file for"
                " each q."
            )
        dAs_criticals.append(dAs_critical)

    cmap = "plasma"
    normalizer = Normalize(
        np.asarray(dAs_criticals).min(), np.asarray(dAs_criticals).max()
    )
    im_map = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    # mean abundances of pollinator species
    fig, axs = plt.subplots(
        nrows=2, ncols=3, figsize=(14, 9), constrained_layout=True, sharex=True,
        sharey=True
    )
    axs_flat = axs.ravel()
    for i, q in enumerate(qs):

        dAs_critical = dAs_criticals[i]
        ax = axs_flat[i]

        extent = [rates.min(), rates.max(), A_init.min(), A_init.max()]
        aspect =  rates.max() / A_init.max()
        ax.set_title(f"$q={q}$")
        im = ax.matshow(
            dAs_critical.T, cmap=cmap, origin="lower", extent=extent, aspect=aspect,
            norm=normalizer
        )
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(
            np.linspace(rates.min(), rates.max(), 6),
            np.linspace(rates.min(), rates.max(), 6)
        )
        ax.set_yticks(
            np.linspace(A_init.min(), A_init.max(), 6),
            np.linspace(A_init.min(), A_init.max(), 6)
        )
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
        if i > 2:
            ax.set_xlabel(r"Rate $\lambda$")
        if i == 0 or i == 3:
            ax.set_ylabel(r"Initial abundance")

    fig.colorbar(im_map, ax=axs.ravel().tolist())
    if save_fig:
        plt.savefig(
            f"figures/plot_abundance_rate_critical_dA_G{G}_nu{nu}.{format}",
            format=f"{format}"
        )


def plot_abundance_rate_critical_dA_all(
    recalculate=False, save_fig=False, format="png", fnumber=0
):

    seed = np.random.SeedSequence().generate_state(1)[0]
    seed = 3892699245
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 25
    N_a = 25
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    G = 1
    nu = 0.9

    qs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rates = np.linspace(0.0001, 0.5, 81)
    A_init = np.linspace(0, 1, 41)
    if recalculate:

        AM = pc.AdaptiveModel(
            N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng, G=G,
            nu=nu, q=0, feasible=True, feasible_iters=100
        )
        while not AM.is_feasible:
            print("Network is not feasible, generating a new network...")
            AM = pc.AdaptiveModel(
                N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng, G=G,
                nu=nu, q=0, feasible=True, feasible_iters=100
            )

        for i, q in enumerate(qs):
            print(f"\nCalculating q: {i+1} out of {len(qs)}...")

            # set q parameter, set seed and rng
            AM.q = q
            AM.rng = rng
            AM.seed = seed

            fname = f"output/abundance_rate_critical_dA_all_G{AM.G}_nu{AM.nu}_q{AM.q}_{fnumber}"
            exp.state_space_abundance_rate_critical_dA_all(
                fname, AM, rates=rates, A_init=A_init
            )

    dAs_criticals = []
    for i, q in enumerate(qs):
        fname = f"output/abundance_rate_critical_dA_all_G{G}_nu{nu}_q{q}_{fnumber}"
        try:
            with np.load(fname + ".npz") as sol:
                rates = sol["rates"]
                A_init = sol["A_init"]
                dAs_critical = sol["dAs_critical"]
        except FileNotFoundError:
            print(
                f"File for q = {q} not found. Make sure there exists an output file for"
                " each q."
            )
        print(dAs_critical.shape)
        dAs_criticals.append(dAs_critical)

    cmap = "plasma"
    normalizer = Normalize(
        np.asarray(dAs_criticals).min(), np.asarray(dAs_criticals).max()
    )
    im_map = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    # mean abundances of pollinator species
    fig, axs = plt.subplots(
        nrows=2, ncols=3, figsize=(14, 9), constrained_layout=True, sharex=True,
        sharey=True
    )
    fig.suptitle("Critical drivers of decline $d_A$")
    axs_flat = axs.ravel()
    for i, q in enumerate(qs):

        dAs_critical = dAs_criticals[i]
        ax = axs_flat[i]

        extent = [rates.min(), rates.max(), A_init.min(), A_init.max()]
        aspect =  rates.max() / A_init.max()
        ax.set_title(f"$q={q}$")
        im = ax.matshow(
            dAs_critical.T, cmap=cmap, origin="lower", extent=extent, aspect=aspect,
            norm=normalizer
        )
        if i > 2:
            ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(
            np.linspace(rates.min(), rates.max(), 4),
            np.linspace(rates.min(), rates.max(), 4)
        )
        ax.set_yticks(
            np.linspace(A_init.min(), A_init.max(), 4),
            np.linspace(A_init.min(), A_init.max(), 4)
        )
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
        if i > 2:
            ax.set_xlabel(r"Rate $\lambda$")
        if i == 0 or i == 3:
            ax.set_ylabel(r"Initial abundance")

    fig.colorbar(im_map, ax=axs.ravel().tolist())
    if save_fig:
        plt.savefig(
            f"figures/plot_abundance_rate_critical_dA_all_G{G}_nu{nu}.{format}",
            format=f"{format}"
        )


def plot_bistability_heatmap_diff_q(recalculate=False, save_fig=False, format="png"):

    seed = np.random.SeedSequence().generate_state(1)[0]
    seed = 3892699245
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 25
    N_a = 25
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    G = 1
    nu = 0.8

    qs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    dAs_init = np.linspace(0, 2, 81)
    A_init = np.linspace(0, 1, 41)
    if recalculate:

        AM = pc.AdaptiveModel(
            N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng, G=G,
            nu=nu, q=0, feasible=True, feasible_iters=50
        )

        for i, q in enumerate(qs):
            print(f"\nCalculating q: {i+1} out of {len(qs)}...")

            # set q parameter, set seed and rng
            AM.q = q
            AM.rng = rng
            AM.seed = seed

            fname = f"output/state_space_abundance_dA_G{AM.G}_nu{AM.nu}_q{AM.q}"
            exp.state_space_abundance_dA(fname, AM, dAs_init=dAs_init, A_init=A_init)

    final_abundances = []
    for i, q in enumerate(qs):
        fname = f"output/state_space_abundance_dA_G{G}_nu{nu}_q{q}"
        try:
            with np.load(fname + ".npz") as sol:
                dAs_init = sol["dAs_init"]
                A_init = sol["A_init"]
                final_abundance = sol["final_abundance"]
        except FileNotFoundError:
            print(
                f"File for q = {q} not found. Make sure there exists an output file for"
                " each q."
            )
        final_abundances.append(final_abundance)

    cmap = "plasma"
    # normalizer = Normalize(
    #     np.asarray(final_abundances).min(), np.asarray(final_abundances).max()
    # )
    normalizer = Normalize(
        0, np.asarray(final_abundances).max()
    )
    im_map = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    # mean abundances of pollinator species
    fig, axs = plt.subplots(
        nrows=3, ncols=3, figsize=(10, 9), constrained_layout=True, sharex=True,
        sharey=True
    )
    axs_flat = axs.ravel()
    dA_cutoff = dAs_init.max()
    for i, q in enumerate(qs):

        final_abundance = final_abundances[i]

        if dA_cutoff is not None:
            final_abundance = final_abundance[:int(len(dAs_init)*dA_cutoff/dAs_init.max())][:]
        ax = axs_flat[i]

        aspect =  dA_cutoff / A_init.max()
        extent = [dAs_init.min(), dA_cutoff, A_init.min(), A_init.max()]
        print(final_abundance.min())
        ax.set_title(f"$q={q}$")
        im = ax.matshow(
            final_abundance.T, cmap=cmap, origin="lower", extent=extent, aspect=aspect,
            norm=normalizer
        )
        if i > 5:
            ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(
            np.linspace(dAs_init.min(), dA_cutoff, 4),
            np.linspace(dAs_init.min(), dA_cutoff, 4)
        )
        ax.set_yticks(
            np.linspace(A_init.min(), A_init.max(), 4),
            np.linspace(A_init.min(), A_init.max(), 4)
        )
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
        if i == 7:
            ax.set_xlabel(r"Drivers of decline $d_A$")
        if i == 3:
            ax.set_ylabel(r"Initial abundance per species")

        # plt.colorbar(im, ax=axs)

    fig.colorbar(im_map, ax=axs.ravel().tolist())
    if save_fig:
        plt.savefig(
            f"figures/plot_bistability_heatmap_diff_q_G{G}_nu{nu}.{format}",
            format=f"{format}"
        )


def plot_dA_init_rate_critical_dA(
    recalculate=False, save_fig=False, format="png", fnumber=0,
    relative_dA_critical=False
):
    seed = np.random.SeedSequence().generate_state(1)[0]
    seed = 3892699245
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 25
    N_a = 25
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    G = 1
    nu = 0.8

    qs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rates = np.linspace(0.0001, 0.5, 41)
    dAs_init = np.linspace(0, 4, 31)
    if recalculate:

        AM = pc.AdaptiveModel(
            N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng, G=G,
            nu=nu, q=0, feasible=True, feasible_iters=50
        )

        for i, q in enumerate(qs):
            print(f"\nCalculating q: {i+1} out of {len(qs)}...")

            # set q parameter, set seed and rng
            AM.q = q
            AM.rng = rng
            AM.seed = seed

            fname = f"output/dA_init_rate_critical_dA_G{AM.G}_nu{AM.nu}_q{AM.q}_{fnumber}"
            exp.state_space_rate_dA(
                fname, AM, rates=rates, dAs_init=dAs_init, A_init=1.5
            )

    dAs_criticals = []
    for i, q in enumerate(qs):
        fname = f"output/dA_init_rate_critical_dA_G{G}_nu{nu}_q{q}_{fnumber}"
        try:
            with np.load(fname + ".npz") as sol:
                rates = sol["rates"]
                dAs_init = sol["dAs_init"]
                dAs_critical = sol["dAs_critical"]
                A_init = sol["A_init"]
        except FileNotFoundError:
            print(
                f"File for q = {q} not found. Make sure there exists an output file for"
                " each q."
            )
        if relative_dA_critical:
            dAs_critical = dAs_critical - dAs_init[np.newaxis, :]
        dAs_criticals.append(dAs_critical)

    cmap = "plasma"
    normalizer = Normalize(
        0, np.asarray(dAs_criticals).max()
    )
    im_map = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    # mean abundances of pollinator species
    fig, axs = plt.subplots(
        nrows=2, ncols=3, figsize=(14, 9), constrained_layout=True, sharex=True,
        sharey=True
    )
    if relative_dA_critical:
        title = f"Relative critical drivers of decline $d_A^*$\nInitial abundance $={A_init:.2f}$"
    else:
        title = f"Critical drivers of decline $d_A^*$\nInitial abundance $={A_init:.2f}$"
    fig.suptitle(title)
    axs_flat = axs.ravel()
    for i, q in enumerate(qs):

        dAs_critical = dAs_criticals[i]
        ax = axs_flat[i]

        extent = [rates.min(), rates.max(), dAs_init.min(), dAs_init.max()]
        aspect =  rates.max() / dAs_init.max()
        ax.set_title(f"$q={q}$")
        im = ax.matshow(
            dAs_critical.T, cmap=cmap, origin="lower", extent=extent, aspect=aspect,
            norm=normalizer
        )
        if i > 2:
            ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(
            np.linspace(rates.min(), rates.max(), 4),
            np.linspace(rates.min(), rates.max(), 4)
        )
        ax.set_yticks(
            np.linspace(dAs_init.min(), dAs_init.max(), 4),
            np.linspace(dAs_init.min(), dAs_init.max(), 4)
        )
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
        if i > 2:
            ax.set_xlabel(r"Rate $\lambda$")
        if i == 0 or i == 3:
            ax.set_ylabel(r"Initial $d_A$")

    fig.colorbar(im_map, ax=axs.ravel().tolist())
    if save_fig:
        if relative_dA_critical:
            save_fname = f"figures/dA_init_rate_relative_critical_dA_G{G}_nu{nu}.{format}"
        else:
            save_fname = f"figures/dA_init_rate_critical_dA_G{G}_nu{nu}.{format}"
        plt.savefig(save_fname, format=f"{format}")


def plot_hysteresis(
    recalculate=False, save_fig=False, format="png", fnumber=0, nu=1, q=0, dA_step=0.02,
    seed=None
):

    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]
    # seed = 3892699245
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 15
    N_a = 35
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    G = 1
    q = 0

    fname = f"output/hysteresis_G{G}_nu{nu}_q{q}_{fnumber}"
    dAs = np.linspace(0, 3, int(3/dA_step)+1)
    if recalculate:

        AM = pc.AdaptiveModel (
            N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng,
            G=G, nu=nu, q=q, feasible=True, feasible_iters=100
        )
        while AM.is_feasible:
            print("Network is not feasible, generating a new network...")
            AM = pc.AdaptiveModel (
                N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng,
                G=G, nu=nu, q=q, feasible=True, feasible_iters=100
            )

        exp.hysteresis(fname, AM, dAs=dAs)

    with np.load(fname + ".npz") as sol:
        dAs = sol["dAs"]
        P_sol_forward = sol["P_sol_forward"]
        A_sol_forward = sol["A_sol_forward"]
        P_sol_backward = sol["P_sol_backward"]
        A_sol_backward = sol["A_sol_backward"]
        is_feasible = sol["is_feasible"]

    # calculate mean and std of abundancies
    P_forward_mean = np.mean(P_sol_forward, axis=1)
    A_forward_mean = np.mean(A_sol_forward, axis=1)
    P_backward_mean = np.mean(P_sol_backward, axis=1)
    A_backward_mean = np.mean(A_sol_backward, axis=1)
    P_forward_std = np.std(P_sol_forward, axis=1, ddof=1)
    A_forward_std = np.std(A_sol_forward, axis=1, ddof=1)
    P_backward_std = np.std(P_sol_backward, axis=1, ddof=1)
    A_backward_std = np.std(A_sol_backward, axis=1, ddof=1)

    # maximum value of dA to plot
    dA_cutoff = 1.2

    # mean abundance
    fig, ax = plt.subplots(figsize=(6.6, 5.5), constrained_layout=True)
    ax1_plot, = ax.plot(dAs, A_forward_mean, color="#0072B2", label="Forward trajectory")
    # ax.errorbar(
    #     dAs, A_forward_mean, yerr=A_forward_std, capsize=5, fmt="none", color="#0072B2"
    # )
    ax1_fill = ax.fill_between(
        dAs, A_forward_mean-A_forward_std, A_forward_mean+A_forward_std,
        color="#0072B2", alpha=0.2
    )
    ax.plot(np.flip(dAs), A_backward_mean, color="#D55E00", label="Backward trajectory")
    # ax.errorbar(
    #     np.flip(dAs), A_backward_mean, yerr=A_backward_std, capsize=5, fmt="none",
    #     color="#D55E00"
    # )
    ax.fill_between(
        np.flip(dAs), A_backward_mean-A_backward_std, A_backward_mean+A_backward_std,
        color="#D55E00", alpha=0.2
    )
    ax.set_xlabel(r"Drivers of decline $d_A$")
    ax.set_ylabel(r"Average pollinator abundance")
    dA_limit = min(dA_cutoff, dAs.max())
    ax.set_xlim(0, dA_limit)
    ax.set_ylim(-0.1, 2.2)
    plt.legend()
    if save_fig:
        plt.savefig(f"figures/hysteresis_pollinators_mean_nu{nu}.{format}", format=f"{format}")

    # abundances of each species
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    line_forward, *_ = ax.plot(
        dAs, A_sol_forward, color="#0072B2", linestyle="-"
    )
    line_backward, *_ = ax.plot(
        np.flip(dAs), A_sol_backward, color="#D55E00", linestyle="--"
    )
    ax.set_xlabel(r"Drivers of decline $d_A$")
    ax.set_ylabel(r"Pollinator abundance per species")
    dA_limit = min(dA_cutoff, dAs.max())
    ax.set_xlim(0, dA_limit)
    ax.set_ylim(-0.1, 2.2)
    plt.legend([line_forward, line_backward], ["Forward trajectory", "Backward trajectory"])
    if save_fig:
        plt.savefig(f"figures/hysteresis_pollinators_all_nu{nu}.{format}", format=f"{format}")

    return


def plot_hysteresis_bifurcation_q(
    recalculate=False, save_fig=False, format="png", fnumber=0
):

    seed = np.random.SeedSequence().generate_state(1)[0]
    rng = np.random.default_rng(seed)

    N_p = 15
    N_a = 35
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    G = 1
    nu = 0.8

    qs = np.linspace(0, 1, 11)
    dAs = np.linspace(0, 4, 31)

    if recalculate:

        AM = pc.AdaptiveModel(
            N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng, G=G,
            nu=nu, q=0, feasible=True, feasible_iters=100
        )

        exp.hysteresis_q(AM, dAs, qs, seed, fnumber=fnumber)

    # data analysis
    dAs_all = []
    P_forward_mean_all = []
    P_backward_mean_all = []
    A_forward_mean_all = []
    A_backward_mean_all = []
    is_feasible_all = []
    for i, q in enumerate(qs):
        fname = f"output/hysteresis_G{G}_nu{nu}_q{q}_{fnumber}"
        try:
            with np.load(fname + ".npz") as sol:
                dAs = sol["dAs"]
                P_sol_forward = sol["P_sol_forward"]
                A_sol_forward = sol["A_sol_forward"]
                P_sol_backward = sol["P_sol_backward"]
                A_sol_backward = sol["A_sol_backward"]
                is_feasible = sol["is_feasible"]
        except FileNotFoundError:
            print(
                f"File for q = {q} not found. Make sure there exists an output file for"
                "each q"
            )

        # calculate mean and std of abundancies
        P_forward_mean = np.mean(P_sol_forward, axis=1)
        A_forward_mean = np.mean(A_sol_forward, axis=1)
        P_backward_mean = np.mean(P_sol_backward, axis=1)
        A_backward_mean = np.mean(A_sol_backward, axis=1)

        dAs_all.append(dAs)
        P_forward_mean_all.append(P_forward_mean)
        P_backward_mean_all.append(P_backward_mean)
        A_forward_mean_all.append(A_forward_mean)
        A_backward_mean_all.append(A_backward_mean)
        is_feasible_all.append(is_feasible)

    dAs_all = np.asarray(dAs_all)
    P_forward_mean_all = np.asarray(P_forward_mean_all)
    P_backward_mean_all = np.asarray(P_backward_mean_all)
    A_forward_mean_all = np.asarray(A_forward_mean_all)
    A_backward_mean_all = np.asarray(A_backward_mean_all)

    extinct_threshold = 0.01   # abundance at which we assume all species to be extinct
    tipping_forward = []
    q_forward = []
    tipping_backward = []
    q_backward = []

    for i, q in enumerate(qs):
        # find tipping points
        try:
            ind = (A_forward_mean_all[i] <= extinct_threshold).nonzero()[0][0]
            tipping_forward.append(dAs_all[i][ind])
            q_forward.append(q)
        except IndexError:
            # no inds exist for which holds the condition A_mean <= threshold
            tipping_forward.append(0)
            q_forward.append(q)
        try:
            ind = (A_backward_mean_all[i] <= extinct_threshold).nonzero()[0][-1]
            tipping_backward.append(np.flip(dAs_all[i])[ind])
            q_backward.append(q)
        except IndexError:
            # no inds exist for which holds the condition A_mean <= threshold
            tipping_backward.append(0)
            q_backward.append(q)

    fig, axs = plt.subplots(figsize=(8, 6), constrained_layout=True)
    fig.suptitle(r"Average $d_A$ of pollinator collapse")
    scatter_collapse = axs.scatter(
        q_forward, tipping_forward, marker="x", color="#0072B2", label="Collapse"
    )
    scatter_recovery = axs.scatter(
        q_backward, tipping_backward, marker="^", color="#D55E00", label="Recovery"
    )
    y_min = 0
    y_max = max(max(tipping_backward), max(tipping_forward))
    width = abs(q_forward[1] - q_forward[0])/2
    for i, is_feasible in enumerate(is_feasible_all):
        if is_feasible:
            color = "#009E73"
        else:
            color = "#CC79A7"
        axs.fill_between(
            [q_forward[i]-width, q_forward[i]+width], y_min, y_max, alpha=0.2,
            color=color, linewidth=0.1
        )
    axs.set_xlabel(r"$q$")
    axs.set_ylabel(r"$d_A$")
    legend_elements = [
        scatter_collapse, scatter_recovery,
        Patch(
            facecolor="#009E73", edgecolor="#009E73", linewidth=0.1,
            label="Feasible network", alpha=0.2
        ),
        Patch(
            facecolor="#CC79A7", edgecolor="#CC79A7", linewidth=0.1,
            label="No feasible network", alpha=0.2
        )
    ]
    axs.legend(handles=legend_elements)

    if save_fig:
        plt.savefig(
            f"figures/hysteresis_bifurcation_diff_q_G{G}_nu{nu}.{format}", format=f"{format}"
        )

    return


def plot_hysteresis_diff_q(
    recalculate=False, save_fig=False, format="png", fnumber=0, qs=None, nu=1,
    dA_step=0.02
):

    seed = np.random.SeedSequence().generate_state(1)[0]
    # seed = 3892699245
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 15
    N_a = 35
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    G = 1

    if qs is None:
        qs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dAs = np.linspace(0, 6, int(6/dA_step)+1)
    if recalculate:

        AM = pc.AdaptiveModel(
            N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng, G=G,
            nu=nu, q=0, feasible=True, feasible_iters=100
        )
        while not AM.is_feasible:
            print("Network is not feasible, generating a new network...")
            AM = pc.AdaptiveModel(
                N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng, G=G,
                nu=nu, q=0, feasible=True, feasible_iters=100
            )

        exp.hysteresis_q(AM, dAs, qs, seed, fnumber=fnumber)

    # maximum value of dA to plot
    dA_cutoff = 4

    # mean abundances of pollinator species
    fig, axs = plt.subplots(
        nrows=4, ncols=3, figsize=(12, 16), constrained_layout=True, sharex=True,
        sharey=True
    )
    axs_flat = axs.ravel()
    for i, q in enumerate(qs):
        fname = f"output/hysteresis_G{G}_nu{nu}_q{q}_{fnumber}"
        try:
            with np.load(fname + ".npz") as sol:
                dAs = sol["dAs"]
                P_sol_forward = sol["P_sol_forward"]
                A_sol_forward = sol["A_sol_forward"]
                P_sol_backward = sol["P_sol_backward"]
                A_sol_backward = sol["A_sol_backward"]
                is_feasible = sol["is_feasible"]
        except FileNotFoundError:
            print(
                f"File for q = {q} not found. Make sure there exists an output file for"
                "each q"
            )

        # calculate mean and std of abundancies
        P_forward_mean = np.mean(P_sol_forward, axis=1)
        A_forward_mean = np.mean(A_sol_forward, axis=1)
        P_backward_mean = np.mean(P_sol_backward, axis=1)
        A_backward_mean = np.mean(A_sol_backward, axis=1)
        P_forward_std = np.std(P_sol_forward, axis=1, ddof=1)
        A_forward_std = np.std(A_sol_forward, axis=1, ddof=1)
        P_backward_std = np.std(P_sol_backward, axis=1, ddof=1)
        A_backward_std = np.std(A_sol_backward, axis=1, ddof=1)

        ax = axs_flat[i]
        ax.set_title(f"$q={q}$")
        line_forward, *_ = ax.plot(dAs, A_forward_mean, color="#0072B2")
        ax1_fill = ax.fill_between(
            dAs, A_forward_mean-A_forward_std, A_forward_mean+A_forward_std,
            color="#0072B2", alpha=0.2
        )
        line_backward, *_ = ax.plot(np.flip(dAs), A_backward_mean, color="#D55E00")
        ax.fill_between(
            np.flip(dAs), A_backward_mean-A_backward_std, A_backward_mean+A_backward_std,
            color="#D55E00", alpha=0.2
        )
        # if i == 0 or i == 3:
        #     ax.set_ylabel(r"Average pollinator abundance")
        dA_limit = min(dA_cutoff, dAs.max())
        ax.set_xlim(0, dA_limit)

    # legend will be placed here. Remove all spines except for the xaxis.
    ax = axs_flat[-1]
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(True)
    ax.tick_params(left=False)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(False)

    fig.supxlabel(r"Drivers of decline $d_A$")
    fig.supylabel(r"Average pollinator abundance")
    plt.legend(
        [line_forward, line_backward], ["Forward trajectory", "Backward trajectory"],
        loc="center"
    )
    if save_fig:
        plt.savefig(f"figures/hysteresis_pollinators_diff_q_mean_nu{nu}.{format}", format=f"{format}")

    # abundances of each species
    fig, axs = plt.subplots(
        nrows=4, ncols=3, figsize=(12, 16), constrained_layout=True, sharex=True,
        sharey=True
    )
    axs_flat = axs.ravel()
    for i, q in enumerate(qs):
        fname = f"output/hysteresis_G{G}_nu{nu}_q{q}_{fnumber}"
        try:
            with np.load(fname + ".npz") as sol:
                dAs = sol["dAs"]
                P_sol_forward = sol["P_sol_forward"]
                A_sol_forward = sol["A_sol_forward"]
                P_sol_backward = sol["P_sol_backward"]
                A_sol_backward = sol["A_sol_backward"]
                is_feasible = sol["is_feasible"]
        except FileNotFoundError:
            print(
                f"File for q = {q} not found. Make sure there exists an output file for"
                "each q"
            )

        ax = axs_flat[i]
        ax.set_title(f"$q={q}$")
        line_forward, *_ = ax.plot(
            dAs, A_sol_forward, color="#0072B2", linestyle="-"
        )
        line_backward, *_ = ax.plot(
            np.flip(dAs), A_sol_backward, color="#D55E00", linestyle="--"
        )
        dA_limit = min(3, dAs.max())
        ax.set_xlim(0, dA_limit)

    # legend will be placed here. Remove all spines except for the xaxis.
    ax = axs_flat[-1]
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(True)
    ax.tick_params(left=False)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(False)

    fig.supxlabel(r"Drivers of decline $d_A$")
    fig.supylabel(r"Pollinator abundance per species")\

    plt.legend(
        [line_forward, line_backward], ["Forward trajectory", "Backward trajectory"],
        loc="center"
    )
    if save_fig:
        plt.savefig(f"figures/hysteresis_pollinators_diff_q_all_nu{nu}.{format}", format=f"{format}")

    print(seed)


def plot_hysteresis_rate(recalculate=False, save_fig=False, format="png", seed=None):

    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]
    rng = np.random.default_rng(seed)

    fname_fixed = f"output/hysteresis_rate_fixed"
    fname_rate = f"output/hysteresis_rate"

    N_p = 15
    N_a = 35
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    G = 1
    nu = 1
    q = 0
    dA_max = 2
    rates = [0.01, 0.1]

    recalculate = True

    if recalculate:

        AM = pc.AdaptiveModel(
            N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng, G=G,
            nu=nu, q=q, feasible=True, feasible_iters=50
        )

        # calculate equilibrium trajctories
        dAs = np.linspace(0, dA_max, 31)
        exp.hysteresis(fname_fixed, AM, dAs=dAs)

        # calculate trajectories for given rates
        exp.hysteresis_rate(
            fname_rate + "0", AM, rate=rates[0], dA_max=dA_max
        )
        exp.hysteresis_rate(
            fname_rate + "1", AM, rate=rates[1], dA_max=dA_max
        )

    with np.load(fname_fixed + ".npz") as sol:
        dAs = sol["dAs"]
        P_sol_forward = sol["P_sol_forward"]
        A_sol_forward = sol["A_sol_forward"]
        P_sol_backward = sol["P_sol_backward"]
        A_sol_backward = sol["A_sol_backward"]
        is_feasible = sol["is_feasible"]

    with np.load(fname_rate + "0" + ".npz") as sol:
        dAs_forward_0 = sol["dAs_forward"]
        dAs_backward_0 = sol["dAs_backward"]
        P_sol_forward_0 = sol["P_sol_forward"]
        A_sol_forward_0 = sol["A_sol_forward"]
        P_sol_backward_0 = sol["P_sol_backward"]
        A_sol_backward_0 = sol["A_sol_backward"]
        is_feasible_0 = sol["is_feasible"]
        rate_0 = sol["rate"]

    with np.load(fname_rate + "1" + ".npz") as sol:
        dAs_forward_1 = sol["dAs_forward"]
        dAs_backward_1 = sol["dAs_backward"]
        P_sol_forward_1 = sol["P_sol_forward"]
        A_sol_forward_1 = sol["A_sol_forward"]
        P_sol_backward_1 = sol["P_sol_backward"]
        A_sol_backward_1 = sol["A_sol_backward"]
        is_feasible_1 = sol["is_feasible"]
        rate_1 = sol["rate"]

    # calculate mean and std of abundancies
    P_forward_mean = np.mean(P_sol_forward, axis=1)
    A_forward_mean = np.mean(A_sol_forward, axis=1)
    P_backward_mean = np.mean(P_sol_backward, axis=1)
    A_backward_mean = np.mean(A_sol_backward, axis=1)
    P_forward_std = np.std(P_sol_forward, axis=1, ddof=1)
    A_forward_std = np.std(A_sol_forward, axis=1, ddof=1)
    P_backward_std = np.std(P_sol_backward, axis=1, ddof=1)
    A_backward_std = np.std(A_sol_backward, axis=1, ddof=1)

    P_forward_mean_0 = np.mean(P_sol_forward_0, axis=1)
    A_forward_mean_0 = np.mean(A_sol_forward_0, axis=1)
    P_backward_mean_0 = np.mean(P_sol_backward_0, axis=1)
    A_backward_mean_0 = np.mean(A_sol_backward_0, axis=1)
    P_forward_std_0 = np.std(P_sol_forward_0, axis=1, ddof=1)
    A_forward_std_0 = np.std(A_sol_forward_0, axis=1, ddof=1)
    P_backward_std_0 = np.std(P_sol_backward_0, axis=1, ddof=1)
    A_backward_std_0 = np.std(A_sol_backward_0, axis=1, ddof=1)

    P_forward_mean_1 = np.mean(P_sol_forward_1, axis=1)
    A_forward_mean_1 = np.mean(A_sol_forward_1, axis=1)
    P_backward_mean_1 = np.mean(P_sol_backward_1, axis=1)
    A_backward_mean_1 = np.mean(A_sol_backward_1, axis=1)
    P_forward_std_1 = np.std(P_sol_forward_1, axis=1, ddof=1)
    A_forward_std_1 = np.std(A_sol_forward_1, axis=1, ddof=1)
    P_backward_std_1 = np.std(P_sol_backward_1, axis=1, ddof=1)
    A_backward_std_1 = np.std(A_sol_backward_1, axis=1, ddof=1)

    # mean abundance
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.plot(dAs, A_forward_mean, color="#0072B2", label="Equilibrium")
    ax.fill_between(
        dAs, A_forward_mean-A_forward_std, A_forward_mean+A_forward_std,
        color="#0072B2", alpha=0.2
    )

    ax.plot(
        dAs_forward_0, A_forward_mean_0, color="#009E73", label=f"$\lambda = {rate_0}$"
    )
    ax.fill_between(
        dAs_forward_0, A_forward_mean_0-A_forward_std_0, A_forward_mean_0+A_forward_std_0,
        color="#009E73", alpha=0.2
    )

    ax.plot(
        dAs_forward_1, A_forward_mean_1, color="#F0E442", label=f"$\lambda = {rate_1}$"
    )
    ax.fill_between(
        dAs_forward_1, A_forward_mean_1-A_forward_std_1, A_forward_mean_1+A_forward_std_1,
        color="#F0E442", alpha=0.2
    )

    ax.set_xlabel(r"Drivers of decline $d_A$")
    ax.set_ylabel(r"Average pollinator abundance")
    plt.legend()
    if save_fig:
        plt.savefig(f"figures/hysteresis_rate_pollinators_forward_mean.{format}", format=f"{format}")

    # mean abundance
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.plot(
        np.flip(dAs), A_backward_mean, color="#0072B2", label="Equilibrium"
    )
    ax.fill_between(
        np.flip(dAs), A_backward_mean-A_backward_std, A_backward_mean+A_backward_std,
        color="#0072B2", alpha=0.2
    )

    ax.plot(
        dAs_backward_0, A_backward_mean_0, color="#009E73", label=f"$\lambda = {rate_0}$"
    )
    ax.fill_between(
        dAs_backward_0, A_backward_mean_0-A_backward_std_0, A_backward_mean_0+A_backward_std_0,
        color="#009E73", alpha=0.2
    )

    ax.plot(
        dAs_backward_1, A_backward_mean_1, color="#F0E442", label=f"$\lambda = {rate_1}$"
    )
    ax.fill_between(
        dAs_backward_1, A_backward_mean_1-A_backward_std_1, A_backward_mean_1+A_backward_std_1,
        color="#F0E442", alpha=0.2
    )

    ax.set_xlabel(r"Drivers of decline $d_A$")
    ax.set_ylabel(r"Average pollinator abundance")
    plt.legend()
    if save_fig:
        plt.savefig(f"figures/hysteresis_rate_pollinators_backward_mean.{format}", format=f"{format}")

    # abundances of each species
    # fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    # line_forward, *_ = ax.plot(
    #     dAs, A_sol_forward, color="#0072B2", linestyle="-"
    # )
    # line_backward, *_ = ax.plot(
    #     np.flip(dAs), A_sol_backward, color="#D55E00", linestyle="--"
    # )
    # ax.set_xlabel(r"Drivers of decline $d_A$")
    # ax.set_ylabel(r"Pollinator abundance per species")
    # plt.legend([line_forward, line_backward], ["Forward trajectory", "Backward trajectory"])
    # plt.savefig(f"figures/hysteresis_rate_pollinators_all.png", format="png")

    return


def plot_phi_q(save_fig=False, format="png"):

    x = np.linspace(0.0001, 2, 100)
    y = lambda x, q: 1/x**q
    qs = np.linspace(0, 1, 6)
    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle("Supply-demand ratio as function of $q$")
    for q in qs:
        ax.plot(x, y(x, q), label=f"$q={q:.1f}$")
    ax.set_xlabel(r"$\sum_j \alpha_{ij} \beta_{ij} A_j$")
    ax.set_ylabel(
        r"$\frac{\phi_i}{P_i} = \frac{1}{\left(\sum_j \alpha_{ij} \beta_{ij} A_j\right)^q}$"
    )
    ax.set_yscale("log")
    plt.legend()
    if save_fig:
        plt.savefig(f"figures/phi.{format}", format=f"{format}")


def time_series(dA=0, save_fig=False, format="png"):

    seed = np.random.SeedSequence().generate_state(1)[0]
    seed = 3627416616
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 15
    N_a = 35
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6

    t_remove = None
    show_extinct_threshold = False
    extinct_threshold = 0.01

    AM = pc.AdaptiveModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type="nested", rng=rng,
        seed=seed, nu=1, G=1, q=0, feasible=True, feasible_iters=100
    )

    # def dA_rate(t, rate):
    #     return rate * t
    # dA = {
    #     "func": dA_rate,
    #     "args": (0.1, )
    # }
    t_end, n_steps = 250, 1000
    sol = AM.solve(t_end, stop_on_equilibrium=True, dA=dA, n_steps=n_steps)

    if t_remove is not None:
        ind_remove = int(t_remove * AM.t.shape[0])
        AM.t = AM.t[ind_remove:]
        AM.y = AM.y[:, ind_remove:]

    fig, (ax1, ax2) = plt.subplots(
        figsize=(9, 4.5), ncols=2, constrained_layout=True
    )

    for i in range(AM.N_p):
        line_plants, = ax1.plot(AM.t, AM.y[i], linewidth=1.5)
    for i in range(AM.N_p, AM.N):
        line_polls, = ax2.plot(AM.t, AM.y[i], linewidth=1.5)

    # for i in range(AM.N_p):
    #     line_plants, = ax1.plot(AM.t, AM.y[i], linewidth=1.25, linestyle="-")
    # for i in range(AM.N_p, AM.N):
    #     line_polls, = ax2.plot(AM.t, AM.y[i], linewidth=1.25, linestyle="-")

    # ax1.set_title("(a)")
    ax1.set_xlabel("Time [-]")
    ax1.set_ylabel("Plant abundance")
    ax1.set_xlim(AM.t.min(), 90)
    ax1.set_ylim(0, AM.y.max()+0.1)
    # ax1.set_xticks([15, 30, 45, 60, 75])
    ax1.set_yticks([0, 0.5, 1, 1.5, 2.0, 2.5])

    # ax2.set_title("(b)")
    ax2.set_xlabel("Time [-]")
    ax2.set_ylabel("Pollinator abundance")
    ax2.set_xlim(AM.t.min(), 90)
    ax2.set_ylim(0, AM.y.max()+0.1)
    # ax2.set_xticks([15, 30, 45, 60, 75])
    ax2.set_yticks([0, 0.5, 1, 1.5, 2.0, 2.5])

    if show_extinct_threshold:
        ax1.hlines(
            extinct_threshold, AM.t.min(), AM.t.max(), colors="black",
            linestyles="dashed", linewidth=2
        )
        ax2.hlines(
            extinct_threshold, AM.t.min(), AM.t.max(), colors="black",
            linestyles="dashed", linewidth=2
        )

    if save_fig:
        plt.savefig(f"figures/time_series_dA{dA}.{format}", format=f"{format}")


def abundance_rate_dependence(
    recalculate=False, save_fig=False, format="png", seed=None, fnumber=0, q=0, nu=1,
    plot_legend=False
):

    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 15
    N_a = 35
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6

    # rates = np.linspace(0.0001, 0.5, 35, endpoint=False)
    # rates = np.concatenate((rates, np.linspace(0.5, 1, 16)))
    rates = np.linspace(0.0001, 0.5, 31, endpoint=False)
    rates = np.concatenate((rates, np.linspace(0.5, 1, 11)))
    # print(rates[0], rates[-1], rates)
    # plt.figure()
    # plt.scatter(rates, np.full(51, 1))
    # plt.show()
    G = 1

    n_reps = 100
    fname = f"output/abundance_rate_dependence_G{G}_nu{nu}_q{q}_nreps{n_reps}_{fnumber}"

    if recalculate:

        S_init = 0.1

        n_dA_maxs = 3
        A_final_all = np.zeros((n_reps, n_dA_maxs, len(rates), N_a))
        dA_collapse_all = np.zeros((n_reps))
        for rep in range(n_reps):
            print(f"\nRepetition {rep+1} out of {n_reps}...")
            AM = pc.AdaptiveModel(
                N_p, N_a, mu, connectance, forbidden, nestedness, network_type="nested",
                rng=rng, seed=seed, nu=nu, G=G, q=q, feasible=True, feasible_iters=100
            )
            while not AM.is_feasible:
                print("Network is not feasible, generating a new network...")
                AM = pc.AdaptiveModel(
                    N_p, N_a, mu, connectance, forbidden, nestedness, network_type="nested",
                    rng=rng, seed=seed, nu=nu, G=G, q=q, feasible=True, feasible_iters=100
                )

            # find equilibrium solution
            # y0 = np.full(AM.N, 0.2, dtype=float)
            # y0 = np.concatenate((y0, AM.alpha.flatten()))
            y0_equilibrium = AM.equilibrium()
            A_final_base = y0_equilibrium[AM.N_p:AM.N]
            # print(y0_equilibrium)

            # find dA critical for equilibrium base model
            y0 = np.full(AM.N, S_init, dtype=float)
            y0 = np.concatenate((y0, AM.alpha.flatten()))
            dA_collapse = AM.find_dA_collapse(dA_step=0.02, y0=y0)
            dA_collapse_all[rep] = dA_collapse

            # calculate abundances as function of rate of change of dA
            # dA_maxs = np.array([
            #     0, dA_collapse-0.2, dA_collapse-0.1, 0.9*dA_collapse, 0.95*dA_collapse,
            #     dA_collapse
            # ])
            # dA_maxs = np.array([
            #     0.5*dA_collapse, 0.8*dA_collapse, 0.9*dA_collapse, 0.95*dA_collapse
            # ])
            dA_maxs = np.array([
                0.2*dA_collapse, 0.5*dA_collapse, 0.9*dA_collapse
            ])
            assert len(dA_maxs) == n_dA_maxs, "lenght of dA_maxs should equal the variable n_dA_maxs"

            AM.nu = nu
            AM.G = G
            AM.q = q
            t_end = int(1e5)
            n_steps = int(1e5)

            for j, dA_max in enumerate(dA_maxs):
                print(f"\ndA_max: {j+1} out of {len(dA_maxs)}...")
                P_final = np.zeros((len(rates), AM.N_p))
                A_final = np.zeros((len(rates), AM.N_a))
                for i, rate in enumerate(rates):
                    print(f"rate: {i+1} out of {len(rates)}...")

                    dA_dict = {
                        "func": util.dA_rate_max,
                        "args": (rate, dA_max)
                    }

                    # y0 = np.full(AM.N, 0.2, dtype=float)
                    # y0 = np.concatenate((y0, AM.alpha.flatten()))
                    # AM.solve(
                    #     t_end, y0=y0, n_steps=n_steps, dA=dA_dict, save_period=0,
                    #     stop_on_equilibrium=True
                    # )

                    # y0 = np.full(AM.N, 5, dtype=float)
                    y0 = np.full(AM.N, S_init, dtype=float)
                    y0 = np.concatenate((y0, AM.alpha.flatten()))
                    # y0 = np.concatenate((y0, np.array([0])))
                    sol = AM.solve(
                        t_end, y0=y0, n_steps=n_steps, dA=dA_dict,
                        save_period=0, stop_on_equilibrium=True, stop_on_collapse=True
                    )
                    # print(util.dA_rate_max(sol.t[-1], rate, dA_max))
                    P_final[i] = AM.y[:AM.N_p, -1]
                    A_final[i] = AM.y[AM.N_p:AM.N, -1]

                A_final_all[rep, j] = A_final

        np.savez(
            fname, rates=rates, dA_maxs=dA_maxs, A_final_all=A_final_all,
            A_final_base=A_final_base, dA_collapse_all=dA_collapse_all
        )

    try:
        with np.load(fname + ".npz") as sol:
            rates = sol["rates"]
            dA_maxs = sol["dA_maxs"]
            A_final_all= sol["A_final_all"]
            dA_collapse_all = sol["dA_collapse_all"]
    except FileNotFoundError:
        print(
            f"File not found. Make sure there exists an output file: "
            f"{fname}"
        )

    # print(dA_collapse_all)

    fig, ax = plt.subplots(
        figsize=(7, 5.5), constrained_layout=True, nrows=1, ncols=1
    )
    colors = ["#CC79A7", "#009E73", "#D55E00", "#E69F00", "#F0E442"]
    linestyles = ["--", "-.", ":", "--", "-."]
    y_max = np.max(A_final_all)
    ax_lines = []
    for j, dA_max in enumerate(dA_maxs):

        A_final = A_final_all[:, j, :, :]
        # A_final_mean = np.mean(A_final, axis=1)

        # A_final /= A_final[:, 0, :]
        # A_final_all = np.zeros((n_reps, n_dA_maxs, len(rates), N_a))
        A_final_alive = np.sum(A_final > 0.01, axis=2, dtype=float)
        # print(A_final_alive[:, 0])
        for rep in range(n_reps):
            with np.errstate(divide="ignore", invalid="ignore"):
                A_final_alive[rep] = np.nan_to_num(
                    A_final_alive[rep] / A_final_alive[rep, 0], copy=False, nan=0
                )
            # remove any other (close to) inf values due to dividing by float 0.0
            for k in range(len(rates)):
                if A_final_alive[rep, k] > 1e300:
                    A_final_alive[rep, k] = 0

        A_final_alive_mean = np.mean(A_final_alive, axis=0)
        A_final_alive_std = np.std(A_final_alive, axis=0, ddof=1)

        A_final_alive_quartiles_1 = np.percentile(A_final_alive, 25, axis=0)

        A_final_alive_quartiles_3 = np.percentile(A_final_alive, 75, axis=0)

        # print(A_final_alive_quartiles_3)
        # exit(-1)
        color = colors[j % len(colors)]
        linestyle = linestyles[j % len(linestyles)]
        # label=f"$d_A^{{max}} = {dA_max:.2f}$"
        ax_line, = ax.plot(
            rates, A_final_alive_mean, color=color, linestyle=linestyle, linewidth=3.
        )
        ax_lines.append(ax_line)
        ax.fill_between(
            rates, np.min((A_final_alive_quartiles_1, A_final_alive_mean), axis=0),
            A_final_alive_mean, alpha=0.35, color=ax_line.get_color(), linewidth=0
        )
        ax.fill_between(
            rates, A_final_alive_mean,
            np.max((A_final_alive_mean, A_final_alive_quartiles_3), axis=0), alpha=0.35,
            color=ax_line.get_color(), linewidth=0
        )
    ax.set_xlabel("Rate of change $\lambda$")
    ax.set_ylabel("Relative persistence of pollinators")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_ylim(0, y_max*1.2)
    ax.set_xlim(rates[0], 0.6)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(
        [rates[0], 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ["0.0001", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
    )
    if plot_legend:
        plt.legend(
            [ax_lines[0], ax_lines[1], ax_lines[2]],
            [
                r"$d^{\mathrm{\ max}}_A = 0.2\cdot d_A^{\mathrm{\ collapse}}$",
                r"$d^{\mathrm{\ max}}_A = 0.5\cdot d_A^{\mathrm{\ collapse}}$",
                r"$d^{\mathrm{\ max}}_A = 0.9\cdot d_A^{\mathrm{\ collapse}}$"
            ]
        )
    # ax_legend.axis("off")
    if save_fig:
        plt.savefig(
            f"figures/abundance_rate_dependence_G{G}_nu{nu}_q{q}_nreps{n_reps}_{fnumber}.{format}",
            format=f"{format}"
        )

    print(seed)


def plot_rate_critical_dA(
    recalculate=False, save_fig=False, format="png", fnumber=0, seed=None
):

    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]
    rng = np.random.default_rng(seed)

    N_p = 15
    N_a = 35
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    G = 1
    nu = 1
    q = 0

    rates = np.linspace(0.0001, 0.5, 81)
    A_init = [0.1, 1]
    n_reps = 25

    if recalculate:

        for rep in range(n_reps):
            print(f"\nRepetition {rep+1} out of {n_reps}...")
            AM = pc.AdaptiveModel(
                N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng, G=G,
                nu=nu, q=q, feasible=True, feasible_iters=100
            )
            while not AM.is_feasible:
                print("Network is not feasible, generating a new network...")
                AM = pc.AdaptiveModel(
                    N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng, G=G,
                    nu=nu, q=q, feasible=True, feasible_iters=100
                )

            fname = f"output/rate_critical_dA_G{AM.G}_nu{AM.nu}_q{AM.q}_rep{rep}_{fnumber}"
            exp.state_space_rate_critical_dA(
                fname, AM, rates=rates, A_init=A_init
            )

    dAs_criticals = []
    for rep in range(n_reps):
        # rep=5
        fname = f"output/rate_critical_dA_G{G}_nu{nu}_q{q}_rep{rep}_{fnumber}"
        try:
            with np.load(fname + ".npz") as sol:
                rates = sol["rates"]
                A_init = sol["A_init"]
                dAs_critical = sol["dAs_critical"]
        except FileNotFoundError:
            print(
                f"File not found. Make sure there exists an output file"
            )
        dAs_criticals.append(dAs_critical)

    # calculate mean and stddev
    dAs_criticals = np.asarray(dAs_criticals)
    dAs_critical_mean = np.mean(dAs_criticals, axis=0)
    dAs_critical_std = np.std(dAs_criticals, axis=0, ddof=1)

    # mean abundances of pollinator species
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(8, 6), constrained_layout=True
    )
    # fig.suptitle(
    #     "Critical value of drivers of decline $d_A$\n"
    #     "at which all pollinators are exinct\n"
    #     "as a function of rate of change of $d_A$"
    # )
    for j, abundance in enumerate(A_init):
        ax.plot(rates, dAs_critical_mean[:, j], label=f"$A_{{init}}={abundance}$")
        if n_reps > 1:
            ax.fill_between(
                rates, dAs_critical_mean[:, j]-dAs_critical_std[:, j],
                dAs_critical_mean[:, j]+dAs_critical_std[:, j], alpha=0.35
            )
    # ax.set_xticks(
    #     np.linspace(rates.min(), rates.max(), 4),
    #     np.linspace(rates.min(), rates.max(), 4)
    # )
    # ax.set_yticks(
    #     np.linspace(A_init.min(), A_init.max(), 4),
    #     np.linspace(A_init.min(), A_init.max(), 4)
    # )
    # ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    # ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    ax.set_xlabel(r"Rate $\lambda$")
    ax.set_ylabel(r"Critical $d_A$ of collapse")
    ax.legend(loc="lower right")

    if save_fig:
        plt.savefig(
            f"figures/plot_rate_critical_dA_G{G}_nu{nu}_nreps{n_reps}_Ainit{A_init}.{format}",
            format=f"{format}"
        )


def plot_rate_critical_dA_diff_q(
    recalculate=False, save_fig=False, format="png", fnumber=0, seed=None, A_init=[0.1],
    qs=[0, 0.1, 0.2, 0.3, 0.4, 0.5], nu=1
):
    plt.rcParams.update({'font.size': 20})

    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 15
    N_a = 35
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    G = 1

    rates = np.linspace(0.0001, 0.5, 121)
    n_reps = 100

    if recalculate:

        for q in qs:
            print(f"\nSolving for q = {q} out of {qs}...")
            for rep in range(n_reps):
                print(f"\nRepetition {rep+1} out of {n_reps}...")
                AM = pc.AdaptiveModel(
                    N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng, G=G,
                    nu=nu, q=q, feasible=True, feasible_iters=100
                )
                while not AM.is_feasible:
                    print("Network is not feasible, generating a new network...")
                    AM = pc.AdaptiveModel(
                        N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng, G=G,
                        nu=nu, q=q, feasible=True, feasible_iters=100
                    )

                fname = f"output/rate_critical_dA_G{AM.G}_nu{AM.nu}_q{AM.q}_rep{rep}_{fnumber}"
                exp.state_space_rate_critical_dA(
                    fname, AM, rates=rates, A_init=A_init
                )

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(6.5, 4.5), constrained_layout=True
    )
    for q in qs:
        dAs_criticals = []
        for rep in range(n_reps):
            # rep=5
            fname = f"output/rate_critical_dA_G{G}_nu{nu}_q{q}_rep{rep}_{fnumber}"
            try:
                with np.load(fname + ".npz") as sol:
                    rates = sol["rates"]
                    A_init = sol["A_init"]
                    dAs_critical = sol["dAs_critical"]
            except FileNotFoundError:
                print(
                    f"File not found. Make sure there exists an output file"
                )
            dAs_criticals.append(dAs_critical)

        # calculate mean and stddev
        dAs_criticals = np.asarray(dAs_criticals)
        dAs_critical_mean = np.mean(dAs_criticals, axis=0)
        dAs_critical_std = np.std(dAs_criticals, axis=0, ddof=1)
        # fig.suptitle(
        #     "Critical value of drivers of decline $d_A$\n"
        #     "at which all pollinators are exinct\n"
        #     "as a function of rate of change of $d_A$"
        # )
        ax.plot(rates, dAs_critical_mean[:, 0], label=f"$q={q}$")
        if n_reps > 1:
            ax.fill_between(
                rates, dAs_critical_mean[:, 0]-dAs_critical_std[:, 0],
                dAs_critical_mean[:, 0]+dAs_critical_std[:, 0], alpha=0.35
            )
    # ax.set_xticks(
    #     np.linspace(rates.min(), rates.max(), 4),
    #     np.linspace(rates.min(), rates.max(), 4)
    # )
    # ax.set_yticks(
    #     np.linspace(A_init.min(), A_init.max(), 4),
    #     np.linspace(A_init.min(), A_init.max(), 4)
    # )
    # ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    # ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    ax.set_ylim(0.7, 5.2)
    ax.set_xlim(rates[0], rates[-1])
    ax.set_xticks([0.0001, 0.1, 0.2, 0.3, 0.4, 0.5], ["0.0001", "0.1", "0.2", "0.3", "0.4", "0.5"])
    ax.set_xlabel(r"Rate of change $\lambda$")
    ax.set_ylabel(r"$d_A^{\mathrm{\ collapse}}$")
    ax.legend(loc="lower right", ncol=len(qs), fontsize=16)

    if save_fig:
        plt.savefig(
            f"figures/plot_rate_critical_dA_diff_q_G{G}_nu{nu}_nreps{n_reps}_Ainit{A_init}.{format}",
            format=f"{format}"
        )

    print(f"Seed used: {seed}")


def plot_bifurcation_feasibility_diff_q(
    recalculate=False, save_fig=False, format="png", fnumber=0, seed=None, qs=None, nu=1
):

    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 15
    N_a = 35
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    G = 1
    if qs is None:
        qs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    n_reps = 100

    if recalculate:

        for q in qs:
            print(f"\nSolving for q = {q} out of {qs}...")
            for rep in range(n_reps):
                print(f"\nRepetition {rep+1} out of {n_reps}...")
                AM = pc.AdaptiveModel(
                    N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng,
                    G=G, nu=nu, q=q, feasible=True, feasible_iters=100
                )

                fname = f"output/dA_collapse_recovery_q{q}_rep{rep}_{fnumber}"
                exp.find_dA_collapse_recovery(fname, AM, dA_step=0.02)

    is_feasible_all = np.zeros((len(qs), n_reps))
    dA_collapse_all = np.zeros((len(qs), n_reps))
    dA_recover_all = np.zeros((len(qs), n_reps))
    for i, q in enumerate(qs):
        for rep in range(n_reps):
            fname = f"output/dA_collapse_recovery_q{q}_rep{rep}_{fnumber}"
            try:
                with np.load(fname + ".npz") as sol:
                    dA_collapse = sol["dA_collapse"]
                    dA_recover = sol["dA_recover"]
                    is_feasible = sol["is_feasible"]
            except FileNotFoundError:
                try:
                    if isinstance(q, float):
                        q = 1
                    elif isinstance(q, int):
                        q = 1.0
                    fname = f"output/dA_collapse_recovery_q{q}_rep{rep}_{fnumber}"
                    with np.load(fname + ".npz") as sol:
                        dA_collapse = sol["dA_collapse"]
                        dA_recover = sol["dA_recover"]
                        is_feasible = sol["is_feasible"]
                except FileNotFoundError:
                    print(
                        f"File not found. Make sure there exists an output file: "
                        f"{fname}"
                    )
            dA_collapse_all[i, rep] = dA_collapse
            dA_recover_all[i, rep] = dA_recover
            if is_feasible:
                is_feasible_all[i, rep] = 1

    # calculate mean and stddev
    dA_collapse_mean = np.mean(dA_collapse_all, axis=1)
    dA_collapse_std = np.std(dA_collapse_all, axis=1, ddof=1)
    dA_recover_mean = np.mean(dA_recover_all, axis=1)
    dA_recover_std = np.std(dA_recover_all, axis=1, ddof=1)
    is_feasible_all = is_feasible_all.sum(axis=1) / n_reps

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(7, 4.5), constrained_layout=True
    )
    # fig, ax = plt.subplots(
    #     nrows=1, ncols=1, figsize=(8, 6), constrained_layout=True
    # )
    # ax.set_prop_cycle(marker_cycler)
    # ax.plot(qs, dA_collapse_mean, label=f"Collapse", color="#D55E00", marker="x")
    ax.errorbar(
        qs, dA_collapse_mean, dA_collapse_std, label=f"Collapse", ecolor="#0072B2",
        marker="s", linestyle="", color="#0072B2", markersize=5
    )
    ax.errorbar(
        qs, dA_recover_mean, dA_recover_std, label=f"Recovery", ecolor="#D55E00",
        marker="^", linestyle="", color="#D55E00", markersize=5
        )
    # ax.plot(qs, dA_recover_mean, label=f"Recovery", color="#0072B2", marker="^")
    # ax.set_xticks(
    #     np.linspace(rates.min(), rates.max(), 4),
    #     np.linspace(rates.min(), rates.max(), 4)
    # )
    # ax.set_yticks(
    #     np.linspace(A_init.min(), A_init.max(), 4),
    #     np.linspace(A_init.min(), A_init.max(), 4)
    # )
    # ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    # ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    # ax.set_ylim(0.7, 4.7)
    ax.set_xlabel(r"Resource competition strength $q$")
    ax.set_ylabel(r"Drivers of decline $d_A$")

    ax2 = ax.twinx()
    ax2.plot(qs, is_feasible_all, label=f"Feasible", color="#009E73", linestyle="dashed")
    ax2.set_ylabel(r"Fraction feasible networks")
    ax2.set_ylim(-0.05, 1.05)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#009E73")
    ax2.yaxis.label.set_color("#009E73")
    ax2.tick_params(axis="y", colors="#009E73", which="both")
    ax.legend(loc="lower right")

    if save_fig:
        plt.savefig(
            f"figures/plot_bifurcation_feasibility_diff_q_G{G}_nu{nu}_nreps{n_reps}.{format}",
            format=f"{format}"
        )

    print(f"Seed used: {seed}")


def plot_dA_rate_max(format="png"):

    rate = 0.1
    dA_max = 1

    ts = np.linspace(0, 15, 100)
    y = [util.dA_rate_max(t, rate, dA_max) for t in ts]

    fig, ax = plt.subplots(figsize=(3.6, 2.5), constrained_layout=True)
    # fig.suptitle("Supply-demand ratio as function of $q$")
    ax.plot(ts, y, color="black")
    ax.hlines(dA_max, 0, dA_max/rate, linestyles="dashed", colors="#E69F00", zorder=-2)
    ax.text(
        5.8, 0.42, f"$\lambda$", color="black", rotation="0"
    )
    ax.set_xlabel("Time [-]")
    ax.set_ylabel(r"$d_A$")
    ax.set_yticks([0, dA_max], ["0",  r"$\bf{d_A^{\ \mathrm{\bf{max}}}}$"], color="#E69F00")
    ax.set_xticks([0, 10], ["0",  "10"], color="black")
    my_colors = ["black", "#E69F00"]
    for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), my_colors):
        ticklabel.set_color(tickcolor)
    # ax.set_xlabels()
    # plt.legend()
    plt.savefig(f"figures/dA_rate_max.{format}", format=f"{format}")


def ms_da_rate(save_fig=False, format="png"):

    ts = np.linspace(0, 25, 100)

    rate_slow = 0.05
    rate_fast = 0.2

    low_max = 0.4
    high_max = 0.8
    full_max = 1

    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

    ax.hlines(full_max, 0, ts.max(), linestyles="solid", colors="black")
    ax.text(5, 1.05, "Full equilibrium collapse", color="black")

    y = [util.dA_rate_max(t, rate_fast, high_max) for t in ts]
    ax.plot(ts, y, linestyle="solid", color="#D55E00")
    ax.text(17, 0.7, "High max", color="#D55E00")

    y = [util.dA_rate_max(t, rate_slow, high_max) for t in ts]
    ax.plot(ts, y, linestyle="dashed", color="#D55E00")

    y = [util.dA_rate_max(t, rate_fast, low_max) for t in ts]
    ax.plot(ts, y, linestyle="solid", color="#CC79A7")
    ax.text(17, 0.3, "Low max", color="#CC79A7")

    y = [util.dA_rate_max(t, rate_slow, low_max) for t in ts]
    ax.plot(ts, y, linestyle="dashed", color="#CC79A7")

    ax.text(1.55, 0.23, "fast increase", color="black", rotation=70)
    ax.text(6.3, 0.23, "slow increase", color="black", rotation=34)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Driver of decline")
    ax.set_xlim(0, ts.max())
    ax.set_ylim(0, full_max+0.2)


    # ax.hlines(dA_max, 0, dA_max/rate, linestyles="dashed", colors="#E69F00", zorder=-2)
    # ax.text(
    #     5.8, 0.42, f"$\lambda$", color="black", rotation="0"
    # )
    # ax.set_xlabel("Time [-]")
    # ax.set_ylabel(r"$d_A$")
    # ax.set_yticks([0, dA_max], ["0",  r"$\bf{d_A^{\ \mathrm{\bf{max}}}}$"], color="#E69F00")
    # ax.set_xticks([0, 10], ["0",  "10"], color="black")
    # my_colors = ["black", "#E69F00"]
    # for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), my_colors):
    #     ticklabel.set_color(tickcolor)

    if save_fig:
        plt.savefig(f"figures/ms/dA_rate_max.{format}", format=f"{format}")

    return


def ms_abundance_rate_dependence(
    recalculate=False, save_fig=False, format="png", seed=None, fnumber=0, q=0, nu=1,
    plot_legend=False
):

    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 15
    N_a = 35
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6

    # rates = np.linspace(0.0001, 0.5, 35, endpoint=False)
    # rates = np.concatenate((rates, np.linspace(0.5, 1, 16)))
    rates = np.linspace(0.0001, 0.5, 31, endpoint=False)
    rates = np.concatenate((rates, np.linspace(0.5, 1, 11)))
    # print(rates[0], rates[-1], rates)
    # plt.figure()
    # plt.scatter(rates, np.full(51, 1))
    # plt.show()
    G = 1

    n_reps = 100
    fname = f"output/abundance_rate_dependence_G{G}_nu{nu}_q{q}_nreps{n_reps}_{fnumber}"

    if recalculate:

        S_init = 0.1

        n_dA_maxs = 3
        A_final_all = np.zeros((n_reps, n_dA_maxs, len(rates), N_a))
        dA_collapse_all = np.zeros((n_reps))
        for rep in range(n_reps):
            print(f"\nRepetition {rep+1} out of {n_reps}...")
            AM = pc.AdaptiveModel(
                N_p, N_a, mu, connectance, forbidden, nestedness, network_type="nested",
                rng=rng, seed=seed, nu=nu, G=G, q=q, feasible=True, feasible_iters=100
            )
            while not AM.is_feasible:
                print("Network is not feasible, generating a new network...")
                AM = pc.AdaptiveModel(
                    N_p, N_a, mu, connectance, forbidden, nestedness, network_type="nested",
                    rng=rng, seed=seed, nu=nu, G=G, q=q, feasible=True, feasible_iters=100
                )

            # find equilibrium solution
            # y0 = np.full(AM.N, 0.2, dtype=float)
            # y0 = np.concatenate((y0, AM.alpha.flatten()))
            y0_equilibrium = AM.equilibrium()
            A_final_base = y0_equilibrium[AM.N_p:AM.N]
            # print(y0_equilibrium)

            # find dA critical for equilibrium base model
            y0 = np.full(AM.N, S_init, dtype=float)
            y0 = np.concatenate((y0, AM.alpha.flatten()))
            dA_collapse = AM.find_dA_collapse(dA_step=0.02, y0=y0)
            dA_collapse_all[rep] = dA_collapse

            # calculate abundances as function of rate of change of dA
            # dA_maxs = np.array([
            #     0, dA_collapse-0.2, dA_collapse-0.1, 0.9*dA_collapse, 0.95*dA_collapse,
            #     dA_collapse
            # ])
            # dA_maxs = np.array([
            #     0.5*dA_collapse, 0.8*dA_collapse, 0.9*dA_collapse, 0.95*dA_collapse
            # ])
            dA_maxs = np.array([
                0.2*dA_collapse, 0.5*dA_collapse, 0.9*dA_collapse
            ])
            assert len(dA_maxs) == n_dA_maxs, "lenght of dA_maxs should equal the variable n_dA_maxs"

            AM.nu = nu
            AM.G = G
            AM.q = q
            t_end = int(1e5)
            n_steps = int(1e5)

            for j, dA_max in enumerate(dA_maxs):
                print(f"\ndA_max: {j+1} out of {len(dA_maxs)}...")
                P_final = np.zeros((len(rates), AM.N_p))
                A_final = np.zeros((len(rates), AM.N_a))
                for i, rate in enumerate(rates):
                    print(f"rate: {i+1} out of {len(rates)}...")

                    dA_dict = {
                        "func": util.dA_rate_max,
                        "args": (rate, dA_max)
                    }

                    # y0 = np.full(AM.N, 0.2, dtype=float)
                    # y0 = np.concatenate((y0, AM.alpha.flatten()))
                    # AM.solve(
                    #     t_end, y0=y0, n_steps=n_steps, dA=dA_dict, save_period=0,
                    #     stop_on_equilibrium=True
                    # )

                    # y0 = np.full(AM.N, 5, dtype=float)
                    y0 = np.full(AM.N, S_init, dtype=float)
                    y0 = np.concatenate((y0, AM.alpha.flatten()))
                    # y0 = np.concatenate((y0, np.array([0])))
                    sol = AM.solve(
                        t_end, y0=y0, n_steps=n_steps, dA=dA_dict,
                        save_period=0, stop_on_equilibrium=True, stop_on_collapse=True
                    )
                    # print(util.dA_rate_max(sol.t[-1], rate, dA_max))
                    P_final[i] = AM.y[:AM.N_p, -1]
                    A_final[i] = AM.y[AM.N_p:AM.N, -1]

                A_final_all[rep, j] = A_final

        np.savez(
            fname, rates=rates, dA_maxs=dA_maxs, A_final_all=A_final_all,
            A_final_base=A_final_base, dA_collapse_all=dA_collapse_all
        )

    try:
        with np.load(fname + ".npz") as sol:
            # print([sol1 for sol1 in sol])
            rates = sol["rates"]
            dA_maxs = sol["dA_maxs"]
            A_final_all= sol["A_final_all"]
            A_final_base = sol["A_final_base"]
    except FileNotFoundError:
        print(
            f"File not found. Make sure there exists an output file: "
            f"{fname}"
        )

    print(dA_maxs)

    fig, ax = plt.subplots(
        figsize=(7, 5.5), constrained_layout=True, nrows=1, ncols=1
    )
    colors = ["#CC79A7", "#009E73", "#D55E00", "#E69F00", "#F0E442"]
    linestyles = ["--", "-.", ":", "--", "-."]
    y_max = np.max(A_final_all)
    ax_lines = []
    for j, dA_max in enumerate(dA_maxs):

        A_final = A_final_all[:, j, :, :]
        # A_final_mean = np.mean(A_final, axis=1)

        # A_final /= A_final[:, 0, :]
        # A_final_all = np.zeros((n_reps, n_dA_maxs, len(rates), N_a))
        A_final_alive = np.sum(A_final > 0.01, axis=2, dtype=float)
        # print(A_final_alive[:, 0])
        for rep in range(n_reps):
            with np.errstate(divide="ignore", invalid="ignore"):
                A_final_alive[rep] = np.nan_to_num(
                    A_final_alive[rep] / A_final_alive[rep, 0], copy=False, nan=0
                )
            # remove any other (close to) inf values due to dividing by float 0.0
            for k in range(len(rates)):
                if A_final_alive[rep, k] > 1e300:
                    A_final_alive[rep, k] = 0

        A_final_alive_mean = np.mean(A_final_alive, axis=0)
        A_final_alive_std = np.std(A_final_alive, axis=0, ddof=1)

        A_final_alive_quartiles_1 = np.percentile(A_final_alive, 25, axis=0)

        A_final_alive_quartiles_3 = np.percentile(A_final_alive, 75, axis=0)

        # print(A_final_alive_quartiles_3)
        # exit(-1)
        color = colors[j % len(colors)]
        linestyle = linestyles[j % len(linestyles)]
        # label=f"$d_A^{{max}} = {dA_max:.2f}$"
        ax_line, = ax.plot(
            rates, A_final_alive_mean, color=color, linestyle=linestyle, linewidth=3.
        )
        ax_lines.append(ax_line)
        ax.fill_between(
            rates, np.min((A_final_alive_quartiles_1, A_final_alive_mean), axis=0),
            A_final_alive_mean, alpha=0.35, color=ax_line.get_color(), linewidth=0
        )
        ax.fill_between(
            rates, A_final_alive_mean,
            np.max((A_final_alive_mean, A_final_alive_quartiles_3), axis=0), alpha=0.35,
            color=ax_line.get_color(), linewidth=0
        )
    ax.set_xlabel("Rate of change $\lambda$")
    ax.set_ylabel("Relative persistence of pollinators")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_ylim(0, y_max*1.2)
    ax.set_xlim(rates[0], 0.6)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(
        [rates[0], 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ["0.0001", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
    )
    if plot_legend:
        plt.legend(
            [ax_lines[0], ax_lines[1], ax_lines[2]],
            [
                r"$d^{\mathrm{\ max}}_A = 0.2\cdot d_A^{\mathrm{\ collapse}}$",
                r"$d^{\mathrm{\ max}}_A = 0.5\cdot d_A^{\mathrm{\ collapse}}$",
                r"$d^{\mathrm{\ max}}_A = 0.9\cdot d_A^{\mathrm{\ collapse}}$"
            ]
        )
    # ax_legend.axis("off")
    if save_fig:
        plt.savefig(
            f"figures/ms_abundance_rate_dependence_G{G}_nu{nu}_q{q}_nreps{n_reps}_{fnumber}.{format}",
            format=f"{format}"
        )

    print(seed)


def ms_abundance_rate_dependence_inset(
    recalculate=False, save_fig=False, format="png", seed=None, fnumber=0, q=0, nu=1,
    plot_legend=False
):

    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 15
    N_a = 35
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6

    # rates = np.linspace(0.0001, 0.5, 35, endpoint=False)
    # rates = np.concatenate((rates, np.linspace(0.5, 1, 16)))
    rates = np.linspace(0.0001, 0.5, 31, endpoint=False)
    rates = np.concatenate((rates, np.linspace(0.5, 1, 11)))
    # print(rates[0], rates[-1], rates)
    # plt.figure()
    # plt.scatter(rates, np.full(51, 1))
    # plt.show()
    G = 1

    n_reps = 100
    fname = f"output/abundance_rate_dependence_G{G}_nu{nu}_q{q}_nreps{n_reps}_{fnumber}"

    if recalculate:

        S_init = 0.1

        n_dA_maxs = 3
        A_final_all = np.zeros((n_reps, n_dA_maxs, len(rates), N_a))
        dA_collapse_all = np.zeros((n_reps))
        for rep in range(n_reps):
            print(f"\nRepetition {rep+1} out of {n_reps}...")
            AM = pc.AdaptiveModel(
                N_p, N_a, mu, connectance, forbidden, nestedness, network_type="nested",
                rng=rng, seed=seed, nu=nu, G=G, q=q, feasible=True, feasible_iters=100
            )
            while not AM.is_feasible:
                print("Network is not feasible, generating a new network...")
                AM = pc.AdaptiveModel(
                    N_p, N_a, mu, connectance, forbidden, nestedness, network_type="nested",
                    rng=rng, seed=seed, nu=nu, G=G, q=q, feasible=True, feasible_iters=100
                )

            # find equilibrium solution
            # y0 = np.full(AM.N, 0.2, dtype=float)
            # y0 = np.concatenate((y0, AM.alpha.flatten()))
            y0_equilibrium = AM.equilibrium()
            A_final_base = y0_equilibrium[AM.N_p:AM.N]
            # print(y0_equilibrium)

            # find dA critical for equilibrium base model
            y0 = np.full(AM.N, S_init, dtype=float)
            y0 = np.concatenate((y0, AM.alpha.flatten()))
            dA_collapse = AM.find_dA_collapse(dA_step=0.02, y0=y0)
            dA_collapse_all[rep] = dA_collapse

            # calculate abundances as function of rate of change of dA
            # dA_maxs = np.array([
            #     0, dA_collapse-0.2, dA_collapse-0.1, 0.9*dA_collapse, 0.95*dA_collapse,
            #     dA_collapse
            # ])
            # dA_maxs = np.array([
            #     0.5*dA_collapse, 0.8*dA_collapse, 0.9*dA_collapse, 0.95*dA_collapse
            # ])
            dA_maxs = np.array([
                0.2*dA_collapse, 0.5*dA_collapse, 0.9*dA_collapse
            ])
            assert len(dA_maxs) == n_dA_maxs, "lenght of dA_maxs should equal the variable n_dA_maxs"

            AM.nu = nu
            AM.G = G
            AM.q = q
            t_end = int(1e5)
            n_steps = int(1e5)

            for j, dA_max in enumerate(dA_maxs):
                print(f"\ndA_max: {j+1} out of {len(dA_maxs)}...")
                P_final = np.zeros((len(rates), AM.N_p))
                A_final = np.zeros((len(rates), AM.N_a))
                for i, rate in enumerate(rates):
                    print(f"rate: {i+1} out of {len(rates)}...")

                    dA_dict = {
                        "func": util.dA_rate_max,
                        "args": (rate, dA_max)
                    }

                    # y0 = np.full(AM.N, 0.2, dtype=float)
                    # y0 = np.concatenate((y0, AM.alpha.flatten()))
                    # AM.solve(
                    #     t_end, y0=y0, n_steps=n_steps, dA=dA_dict, save_period=0,
                    #     stop_on_equilibrium=True
                    # )

                    # y0 = np.full(AM.N, 5, dtype=float)
                    y0 = np.full(AM.N, S_init, dtype=float)
                    y0 = np.concatenate((y0, AM.alpha.flatten()))
                    # y0 = np.concatenate((y0, np.array([0])))
                    sol = AM.solve(
                        t_end, y0=y0, n_steps=n_steps, dA=dA_dict,
                        save_period=0, stop_on_equilibrium=True, stop_on_collapse=True
                    )
                    # print(util.dA_rate_max(sol.t[-1], rate, dA_max))
                    P_final[i] = AM.y[:AM.N_p, -1]
                    A_final[i] = AM.y[AM.N_p:AM.N, -1]

                A_final_all[rep, j] = A_final

        np.savez(
            fname, rates=rates, dA_maxs=dA_maxs, A_final_all=A_final_all,
            A_final_base=A_final_base, dA_collapse_all=dA_collapse_all
        )

    try:
        with np.load(fname + ".npz") as sol:
            # print([sol1 for sol1 in sol])
            rates = sol["rates"]
            dA_maxs = sol["dA_maxs"]
            A_final_all= sol["A_final_all"]
            A_final_base = sol["A_final_base"]
    except FileNotFoundError:
        print(
            f"File not found. Make sure there exists an output file: "
            f"{fname}"
        )

    print(dA_maxs)

    fig, ax = plt.subplots(
        figsize=(7, 5), constrained_layout=True, nrows=1, ncols=1
    )
    colors = ["#CC79A7", "#009E73", "#D55E00", "#E69F00", "#F0E442"]
    linestyles = ["--", "-.", ":", "--", "-."]
    y_max = np.max(A_final_all)
    ax_lines = []
    for j, dA_max in enumerate(dA_maxs):

        A_final = A_final_all[:, j, :, :]
        # A_final_mean = np.mean(A_final, axis=1)

        # A_final /= A_final[:, 0, :]
        # A_final_all = np.zeros((n_reps, n_dA_maxs, len(rates), N_a))
        A_final_alive = np.sum(A_final > 0.01, axis=2, dtype=float)
        # print(A_final_alive[:, 0])
        for rep in range(n_reps):
            with np.errstate(divide="ignore", invalid="ignore"):
                A_final_alive[rep] = np.nan_to_num(
                    A_final_alive[rep] / A_final_alive[rep, 0], copy=False, nan=0
                )
            # remove any other (close to) inf values due to dividing by float 0.0
            for k in range(len(rates)):
                if A_final_alive[rep, k] > 1e300:
                    A_final_alive[rep, k] = 0

        A_final_alive_mean = np.mean(A_final_alive, axis=0)
        A_final_alive_std = np.std(A_final_alive, axis=0, ddof=1)

        A_final_alive_quartiles_1 = np.percentile(A_final_alive, 25, axis=0)

        A_final_alive_quartiles_3 = np.percentile(A_final_alive, 75, axis=0)

        # print(A_final_alive_quartiles_3)
        # exit(-1)
        color = colors[j % len(colors)]
        linestyle = linestyles[j % len(linestyles)]
        # label=f"$d_A^{{max}} = {dA_max:.2f}$"
        ax_line, = ax.plot(
            rates, A_final_alive_mean, color=color, linestyle=linestyle, linewidth=3.
        )
        ax_lines.append(ax_line)
        ax.fill_between(
            rates, np.min((A_final_alive_quartiles_1, A_final_alive_mean), axis=0),
            A_final_alive_mean, alpha=0.35, color=ax_line.get_color(), linewidth=0
        )
        ax.fill_between(
            rates, A_final_alive_mean,
            np.max((A_final_alive_mean, A_final_alive_quartiles_3), axis=0), alpha=0.35,
            color=ax_line.get_color(), linewidth=0
        )
    ax.set_xlabel("Rate of change $\lambda$")
    ax.set_ylabel("Relative persistence of pollinators")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_ylim(0, y_max*1.2)
    ax.set_xlim(rates[0], 0.6)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(
        [rates[0], 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ["0.0001", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
    )

    # inset
    # Create a set of inset Axes: these should fill the bounding box allocated to
    # them.
    ax2 = fig.add_axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax, [0.45,0.3, 0.55,0.55])
    ax2.set_axes_locator(ip)

    ts = np.linspace(0, 25, 100)

    rate_slow = 0.05
    rate_fast = 0.2

    low_max = 0.4
    med_max = 0.6
    high_max = 0.8
    full_max = 1

    # fig, ax2 = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

    ax2.hlines(full_max, 0, ts.max(), linestyles="solid", colors="black")
    ax2.text(5, 1.05, "Full equilibrium collapse", color="black")

    y = [util.dA_rate_max(t, rate_fast, high_max) for t in ts]
    ax2.plot(ts, y, linestyle="solid", color="#D55E00")
    ax2.text(17, 0.7, "High max", color="#D55E00")

    y = [util.dA_rate_max(t, rate_slow, high_max) for t in ts]
    ax2.plot(ts, y, linestyle="dashed", color="#D55E00")

    y = [util.dA_rate_max(t, rate_fast, med_max) for t in ts]
    ax2.plot(ts, y, linestyle="solid", color="#009E73")
    ax2.text(17, 0.5, "Med max", color="#009E73")

    y = [util.dA_rate_max(t, rate_slow, med_max) for t in ts]
    ax2.plot(ts, y, linestyle="dashed", color="#009E73")

    y = [util.dA_rate_max(t, rate_fast, low_max) for t in ts]
    ax2.plot(ts, y, linestyle="solid", color="#CC79A7")
    ax2.text(17, 0.3, "Low max", color="#CC79A7")

    y = [util.dA_rate_max(t, rate_slow, low_max) for t in ts]
    ax2.plot(ts, y, linestyle="dashed", color="#CC79A7")

    ax2.text(1.55, 0.23, "fast increase", color="black", rotation=70)
    ax2.text(6.3, 0.23, "slow increase", color="black", rotation=37)
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    ax2.set_xlabel("Time")
    ax2.set_ylabel(r"Driver of decline")
    ax2.set_xlim(0, ts.max())
    ax2.set_ylim(0, full_max+0.2)
    ax2.spines.top.set_visible(True)
    ax2.spines.right.set_visible(True)

    # ax_legend.axis("off")
    if save_fig:
        plt.savefig(
            f"figures/ms_inset_abundance_rate_dependence_G{G}_nu{nu}_q{q}_nreps{n_reps}_{fnumber}.{format}",
            format=f"{format}"
        )

    print(seed)


def ms_plot_bifurcation_feasibility_diff_q(
    recalculate=False, save_fig=False, format="png", fnumber=0, seed=None, qs=None, nu=1
):

    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]
    print(seed)
    rng = np.random.default_rng(seed)

    N_p = 15
    N_a = 35
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    G = 1
    if qs is None:
        qs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    n_reps = 100

    if recalculate:

        for q in qs:
            print(f"\nSolving for q = {q} out of {qs}...")
            for rep in range(n_reps):
                print(f"\nRepetition {rep+1} out of {n_reps}...")
                AM = pc.AdaptiveModel(
                    N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed, rng=rng,
                    G=G, nu=nu, q=q, feasible=True, feasible_iters=100
                )

                fname = f"output/dA_collapse_recovery_q{q}_rep{rep}_{fnumber}"
                exp.find_dA_collapse_recovery(fname, AM, dA_step=0.02)

    is_feasible_all = np.zeros((len(qs), n_reps))
    dA_collapse_all = np.zeros((len(qs), n_reps))
    dA_recover_all = np.zeros((len(qs), n_reps))
    for i, q in enumerate(qs):
        for rep in range(n_reps):
            fname = f"output/dA_collapse_recovery_q{q}_rep{rep}_{fnumber}"
            try:
                with np.load(fname + ".npz") as sol:
                    dA_collapse = sol["dA_collapse"]
                    dA_recover = sol["dA_recover"]
                    is_feasible = sol["is_feasible"]
            except FileNotFoundError:
                try:
                    if isinstance(q, float):
                        q = 1
                    elif isinstance(q, int):
                        q = 1.0
                    fname = f"output/dA_collapse_recovery_q{q}_rep{rep}_{fnumber}"
                    with np.load(fname + ".npz") as sol:
                        dA_collapse = sol["dA_collapse"]
                        dA_recover = sol["dA_recover"]
                        is_feasible = sol["is_feasible"]
                except FileNotFoundError:
                    print(
                        f"File not found. Make sure there exists an output file: "
                        f"{fname}"
                    )
            dA_collapse_all[i, rep] = dA_collapse
            dA_recover_all[i, rep] = dA_recover
            if is_feasible:
                is_feasible_all[i, rep] = 1

    # calculate mean and stddev
    dA_collapse_mean = np.mean(dA_collapse_all, axis=1)
    dA_collapse_std = np.std(dA_collapse_all, axis=1, ddof=1)
    dA_recover_mean = np.mean(dA_recover_all, axis=1)
    dA_recover_std = np.std(dA_recover_all, axis=1, ddof=1)
    is_feasible_all = is_feasible_all.sum(axis=1) / n_reps

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(7, 4.5), constrained_layout=True
    )
    # fig, ax = plt.subplots(
    #     nrows=1, ncols=1, figsize=(8, 6), constrained_layout=True
    # )
    # ax.set_prop_cycle(marker_cycler)
    # ax.plot(qs, dA_collapse_mean, label=f"Collapse", color="#D55E00", marker="x")
    ax.errorbar(
        qs, dA_collapse_mean, dA_collapse_std, label=f"Collapse", ecolor="#0072B2",
        marker="s", linestyle="", color="#0072B2", markersize=5
    )
    ax.errorbar(
        qs, dA_recover_mean, dA_recover_std, label=f"Recovery", ecolor="#D55E00",
        marker="^", linestyle="", color="#D55E00", markersize=5
        )
    # ax.plot(qs, dA_recover_mean, label=f"Recovery", color="#0072B2", marker="^")
    # ax.set_xticks(
    #     np.linspace(rates.min(), rates.max(), 4),
    #     np.linspace(rates.min(), rates.max(), 4)
    # )
    # ax.set_yticks(
    #     np.linspace(A_init.min(), A_init.max(), 4),
    #     np.linspace(A_init.min(), A_init.max(), 4)
    # )
    # ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    # ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    # ax.set_ylim(0.7, 4.7)
    ax.set_xlabel(r"Resource competition strength $q$")
    ax.set_ylabel(r"Drivers of decline $d_A$")

    ax2 = ax.twinx()
    ax2.plot(qs, is_feasible_all, label=f"Feasible", color="#009E73", linestyle="dashed")
    ax2.set_ylabel(r"Fraction feasible networks")
    ax2.set_ylim(-0.05, 1.05)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#009E73")
    ax2.yaxis.label.set_color("#009E73")
    ax2.tick_params(axis="y", colors="#009E73", which="both")
    ax.legend(loc="lower right")

    if save_fig:
        plt.savefig(
            f"figures/plot_bifurcation_feasibility_diff_q_G{G}_nu{nu}_nreps{n_reps}.{format}",
            format=f"{format}"
        )

    print(f"Seed used: {seed}")


if __name__ == "__main__":

    # ms_da_rate(save_fig=True, format="png")
    # ms_abundance_rate_dependence(save_fig=False)

    # ms_abundance_rate_dependence_inset(
    #     False, False, format="png", fnumber=5, q=0, nu=1
    # )
    # ms_abundance_rate_dependence_inset(
    #     False, False, format="png", fnumber=5, q=0.2, nu=0.7
    # )
    #
    # ms_abundance_rate_dependence(
    #     False, False, format="png", fnumber=5, q=0.2, nu=0.7
    # )

    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     False, False, format="png", fnumber=0, seed=None, nu=1
    # )
    # print(f"{timer()-t0:.2f} seconds")

    # holling()
    # adjacency_matrix()
    # beta_plot(True, format="png")
    # time_series(dA=0, save_fig=True, format="png")
    # time_series(dA=1, save_fig=True, format="png")

    # plt.show()
    # exit(0)

    # plot_abundance_rate_critical_dA(False, True)
    # plot_abundance_rate_critical_dA_all(False, True, fnumber=0)
    # print("plot_dA_init_rate_critical_dA")
    # plot_dA_init_rate_critical_dA(True, True, fnumber=1, relative_dA_critical=False)
    # plot_bistability_heatmap_diff_q(True, True)
    # plot_hysteresis(recalculate=True, save_fig=True)
    # print("plot_hysteresis_bifurcation_q")
    # plot_hysteresis_bifurcation_q(True, True, fnumber=2)
    # plot_hysteresis_diff_q(True, True)
    # plot_hysteresis_rate(True, True)
    # plot_rate_critical_dA(True, True, fnumber=0, format="pdf")

    # t0 = timer()
    # plot_hysteresis(
    #     False, True, format="png", fnumber=2, nu=1, q=0, dA_step=0.02,
    #     seed=484356304
    # )
    # print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_rate_critical_dA_diff_q(
        False, False, fnumber=0, format="png", qs=[0, 0.2, 0.4], nu=1,
        A_init=[0.1]
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_rate_critical_dA_diff_q(
        False, False, fnumber=1, format="png", qs=[0, 0.2, 0.4], nu=1,
        A_init=[2]
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_rate_critical_dA_diff_q(
        False, False, fnumber=0, format="png", qs=[0, 0.2, 0.4], nu=0.8,
        A_init=[0.1]
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_rate_critical_dA_diff_q(
        False, False, fnumber=1, format="png", qs=[0, 0.2, 0.4], nu=0.8,
        A_init=[2]
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_rate_critical_dA_diff_q(
        False, False, format="png", fnumber=0, qs=[0, 0.2, 0.4], nu=0.7,
        A_init=[0.1]
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_rate_critical_dA_diff_q(
        False, False, format="png", fnumber=1, qs=[0, 0.2, 0.4], nu=0.7,
        A_init=[2]
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_rate_critical_dA_diff_q(
        False, False, format="png", fnumber=0, qs=[0.1, 0.2, 0.4], nu=0.6,
        A_init=[0.1]
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_rate_critical_dA_diff_q(
        False, False, format="png", fnumber=1, qs=[0.1, 0.2, 0.4], nu=0.6,
        A_init=[2]
    )
    print(f"{timer()-t0:.2f} seconds")

    plt.show()
    exit(0)

    # no adaptation
    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     True, True, format="pdf", fnumber=0, seed=None, nu=1, qs=[0, 0.1, 0.2]
    # )
    # print(f"{timer()-t0:.2f} seconds")

    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     True, True, format="pdf", fnumber=0, seed=None, nu=1, qs=[0.3, 0.4, 0.5]
    # )
    # print(f"{timer()-t0:.2f} seconds")

    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     True, True, format="pdf", fnumber=0, seed=None, nu=1, qs=[0.6, 0.7, 0.8]
    # )
    # print(f"{timer()-t0:.2f} seconds")

    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     False, True, format="pdf", fnumber=0, seed=None, nu=1,
    #     qs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # )
    # print(f"{timer()-t0:.2f} seconds")

    # adaptation
    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     True, True, format="pdf", fnumber=0, seed=None, nu=0.8, qs=[0, 0.1, 0.2, 0.3]
    # )
    # print(f"{timer()-t0:.2f} seconds")
    #
    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     True, True, format="pdf", fnumber=0, seed=None, nu=0.8, qs=[0.4, 0.5, 0.6, 0.7]
    # )
    # print(f"{timer()-t0:.2f} seconds")
    #
    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     True, True, format="pdf", fnumber=0, seed=None, nu=0.8, qs=[0.8, 0.9, 1]
    # )
    # print(f"{timer()-t0:.2f} seconds")

    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     True, True, format="pdf", fnumber=3, seed=None, nu=0.6, qs=[0, 0.1]
    # )
    # print(f"{timer()-t0:.2f} seconds")

    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     True, True, format="pdf", fnumber=3, seed=None, nu=0.6, qs=[0.2, 0.3]
    # )
    # print(f"{timer()-t0:.2f} seconds")
    #
    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     True, True, format="pdf", fnumber=3, seed=None, nu=0.6, qs=[0.4, 0.5, 0.6]
    # )
    # print(f"{timer()-t0:.2f} seconds")

    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     True, True, format="pdf", fnumber=3, seed=None, nu=0.6, qs=[0.7, 0.8]
    # )
    # print(f"{timer()-t0:.2f} seconds")

    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     True, True, format="pdf", fnumber=3, seed=None, nu=0.6, qs=[0.9, 1.0]
    # )
    # print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_bifurcation_feasibility_diff_q(
        False, True, format="png", fnumber=0, seed=None, nu=1,
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_bifurcation_feasibility_diff_q(
        False, True, format="png", fnumber=1, seed=None, nu=0.8,
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_bifurcation_feasibility_diff_q(
        False, True, format="png", fnumber=2, seed=None, nu=0.7,
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_bifurcation_feasibility_diff_q(
        False, True, format="png", fnumber=3, seed=None, nu=0.6,
    )
    print(f"{timer()-t0:.2f} seconds")

    # t0 = timer()
    # plot_bifurcation_feasibility_diff_q(
    #     False, True, format="pdf", fnumber=2, seed=None, nu=0.7
    # )
    # print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_hysteresis_diff_q(
        False, True, format="png", fnumber=2, nu=1,
        dA_step=0.02
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_hysteresis_diff_q(
        False, True, format="png", fnumber=2, nu=0.8,
        dA_step=0.02
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_hysteresis_diff_q(
        False, True, format="png", fnumber=2, nu=0.7,
        dA_step=0.02
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    plot_hysteresis_diff_q(
        False, True, format="png", fnumber=2, nu=0.6,
        dA_step=0.02
    )
    print(f"{timer()-t0:.2f} seconds")

    # t0 = timer()
    # abundance_rate_dependence(
    #     True, True, format="pdf", fnumber=3, q=0, nu=1
    # )
    # print(f"{timer()-t0:.2f} seconds")

    # t0 = timer()
    # abundance_rate_dependence(
    #     True, True, format="pdf", fnumber=3, q=0.1, nu=0.8
    # )
    # print(f"{timer()-t0:.2f} seconds")

    # t0 = timer()
    # abundance_rate_dependence(
    #     True, True, format="pdf", fnumber=3, q=0.2, nu=0.7
    # )
    # print(f"{timer()-t0:.2f} seconds")

    # t0 = timer()
    # abundance_rate_dependence(
    #     True, True, format="pdf", fnumber=3, q=0.2, nu=0.6
    # )
    # print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    abundance_rate_dependence(
        False, True, format="png", fnumber=5, q=0, nu=1, plot_legend=True
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    abundance_rate_dependence(
        False, True, format="png", fnumber=5, q=0.1, nu=0.8
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    abundance_rate_dependence(
        False, True, format="png", fnumber=5, q=0.2, nu=0.7
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    abundance_rate_dependence(
        False, True, format="png", fnumber=5, q=0.2, nu=0.6
    )
    print(f"{timer()-t0:.2f} seconds")

    # plot_phi_q(True)
    # abundance_rate_dependence(True, True)
    # plt.show()
