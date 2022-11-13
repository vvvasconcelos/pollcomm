#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 28/05/2022
# ---------------------------------------------------------------------------
""" graphs.py

Script to generate all graphs from manuscript except the sensitivity analysis.
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
plt.rc("font", family="serif", size=10.5)
plt.rc("savefig", dpi=600)
plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
plt.rc("lines", linewidth=2, markersize=7, markeredgewidth=1.8)
plt.rc("axes.spines", top=False, right=False)
# mpl.rcParams['axes.spines.top'] = False


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
    recalculate=False, format="png", seed=None, fnumber=0, q=0, nu=1, ax=None
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

    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(7, 5), constrained_layout=True, nrows=1, ncols=1
        )
        save_fig = True
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
    ax.set_xlabel("rate of change $\lambda$ of driver of decline $d_A$")
    ax.set_ylabel("relative pollinator persistence")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_ylim(0, y_max*1.2)
    ax.set_xlim(rates[0], 0.6)
    ax.set_ylim(-0.1, 1.1)
    # ax.set_xticks(
    #     [rates[0], 0.2, 0.4, 0.6, 0.8, 1.0],
    #     [r"$10^{-4}$", "0.2", "0.4", "0.6", "0.8", "1.0"]
    # )
    ax.set_xticks(
        [rates[0], 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [r"$10^{-4}$", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
    )
    # if plot_legend:
    #     plt.legend(
    #         [ax_lines[0], ax_lines[1], ax_lines[2]],
    #         [
    #             r"$d^{\mathrm{\ max}}_A = 0.2\cdot d_A^{\mathrm{\ collapse}}$",
    #             r"$d^{\mathrm{\ max}}_A = 0.5\cdot d_A^{\mathrm{\ collapse}}$",
    #             r"$d^{\mathrm{\ max}}_A = 0.9\cdot d_A^{\mathrm{\ collapse}}$"
    #         ]
    #     )
    # ax_legend.axis("off")
    if save_fig:
        plt.savefig(
            f"figures/ms/ms_abundance_rate_dependence_G{G}_nu{nu}_q{q}_nreps{n_reps}_{fnumber}.{format}",
            format=f"{format}"
        )

    print(seed)


def ms_abundance_rate_dependence_inset(
    recalculate=False, format="png", seed=None, fnumber=0, q=0, nu=1, ax=None, fig=None
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

    save_fig = False
    if ax is None or fig is None:
        fig, ax = plt.subplots(
            figsize=(7, 5), constrained_layout=True, nrows=1, ncols=1
        )
        save_fig = True
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
    ax.set_xlabel("rate of change $\lambda$ of driver of decline $d_A$")
    ax.set_ylabel("relative pollinator persistence")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_ylim(0, y_max*1.2)
    ax.set_xlim(rates[0], 0.6)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(
        [rates[0], 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [r"$10^{-4}$", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
    )

    # inset
    # Create a set of inset Axes: these should fill the bounding box allocated to
    # them.
    ax2 = fig.add_axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax, [0.4,0.3, 0.62,0.55])
    ax2.set_axes_locator(ip)

    ts = np.linspace(0, 25, 100)

    rate_slow = 0.05
    rate_fast = 0.2

    low_max = 0.2
    med_max = 0.5
    high_max = 0.9
    full_max = 1

    # fig, ax2 = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

    fontsize = 9
    ax2.hlines(full_max, 0, ts.max(), linestyles="solid", colors="black")
    ax2.text(5, 1.05, r"$d_A^{\mathrm{\ collapse}}$ ($\theta=1$)", color="black", fontsize=fontsize)

    y = [util.dA_rate_max(t, rate_fast, high_max) for t in ts]
    ax2.plot(ts, y, linestyle="solid", color="#D55E00")
    ax2.text(18, 0.8, r"$\theta=0.9$", color="#D55E00", fontsize=fontsize)

    y = [util.dA_rate_max(t, rate_slow, high_max) for t in ts]
    ax2.plot(ts, y, linestyle="dashed", color="#D55E00")

    y = [util.dA_rate_max(t, rate_fast, med_max) for t in ts]
    ax2.plot(ts, y, linestyle="solid", color="#009E73")
    ax2.text(18, 0.4, r"$\theta=0.5$", color="#009E73", fontsize=fontsize)

    y = [util.dA_rate_max(t, rate_slow, med_max) for t in ts]
    ax2.plot(ts, y, linestyle="dashed", color="#009E73")

    y = [util.dA_rate_max(t, rate_fast, low_max) for t in ts]
    ax2.plot(ts, y, linestyle="solid", color="#CC79A7")
    ax2.text(18, 0.1, r"$\theta=0.2$", color="#CC79A7", fontsize=fontsize)

    y = [util.dA_rate_max(t, rate_slow, low_max) for t in ts]
    ax2.plot(ts, y, linestyle="dashed", color="#CC79A7")

    ax2.text(1.75, 0.25, "fast increase", color="black", rotation=76, fontsize=fontsize)
    ax2.text(6, 0.2, "slow increase", color="black", rotation=45, fontsize=fontsize)
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    ax2.set_xlabel("time", fontsize=fontsize)
    ax2.set_ylabel(r"driver of decline $d_A$", fontsize=fontsize)
    ax2.set_xlim(0, ts.max())
    ax2.set_ylim(0, full_max+0.2)
    ax2.spines.top.set_visible(True)
    ax2.spines.right.set_visible(True)

    # fontsize = 9
    # ax2.hlines(full_max, 0, ts.max(), linestyles="solid", colors="black")
    # ax2.text(5, 1.05, "collapse ($d_A^*$)", color="black", fontsize=fontsize)
    #
    # y = [util.dA_rate_max(t, rate_fast, high_max) for t in ts]
    # ax2.plot(ts, y, linestyle="solid", color="#D55E00")
    # ax2.text(18, 0.78, "90% $d_A^*$", color="#D55E00", fontsize=fontsize)
    #
    # y = [util.dA_rate_max(t, rate_slow, high_max) for t in ts]
    # ax2.plot(ts, y, linestyle="dashed", color="#D55E00")
    #
    # y = [util.dA_rate_max(t, rate_fast, med_max) for t in ts]
    # ax2.plot(ts, y, linestyle="solid", color="#009E73")
    # ax2.text(18, 0.38, "50% $d_A^*$", color="#009E73", fontsize=fontsize)
    #
    # y = [util.dA_rate_max(t, rate_slow, med_max) for t in ts]
    # ax2.plot(ts, y, linestyle="dashed", color="#009E73")
    #
    # y = [util.dA_rate_max(t, rate_fast, low_max) for t in ts]
    # ax2.plot(ts, y, linestyle="solid", color="#CC79A7")
    # ax2.text(18, 0.08, "20% $d_A^*$", color="#CC79A7", fontsize=fontsize)
    #
    # y = [util.dA_rate_max(t, rate_slow, low_max) for t in ts]
    # ax2.plot(ts, y, linestyle="dashed", color="#CC79A7")
    #
    # ax2.text(1.75, 0.25, "fast increase", color="black", rotation=76, fontsize=fontsize)
    # ax2.text(6, 0.2, "slow increase", color="black", rotation=45, fontsize=fontsize)
    # ax2.get_xaxis().set_ticks([])
    # ax2.get_yaxis().set_ticks([])
    # ax2.set_xlabel("time", fontsize=fontsize)
    # ax2.set_ylabel(r"driver of decline $d_A$", fontsize=fontsize)
    # ax2.set_xlim(0, ts.max())
    # ax2.set_ylim(0, full_max+0.2)
    # ax2.spines.top.set_visible(True)
    # ax2.spines.right.set_visible(True)

    # ax_legend.axis("off")
    if save_fig:
        plt.savefig(
            f"figures/ms/ms_inset_abundance_rate_dependence_G{G}_nu{nu}_q{q}_nreps{n_reps}_{fnumber}.{format}",
            format=f"{format}"
        )

    print(seed)


def ms_bifurcation_feasibility_diff_q(
    recalculate=False, format="png", fnumber=0, seed=None, qs=None, nu=1, ax=None
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

    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(3.5, 2.5), constrained_layout=True
        )
        save_fig = True

    # ax.set_prop_cycle(marker_cycler)
    # ax.plot(qs, dA_collapse_mean, label=f"Collapse", color="#D55E00", marker="x")
    ax.errorbar(
        qs, dA_collapse_mean, dA_collapse_std, label=f"collapse", ecolor="#0072B2",
        marker="s", linestyle="", color="#0072B2", markersize=5
    )
    ax.errorbar(
        qs, dA_recover_mean, dA_recover_std, label=f"recovery", ecolor="#D55E00",
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
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.3, 6.7)
    ax.set_xlabel(r"resource congestion $q$")
    ax.set_ylabel(r"driver of decline $d_A$")

    ax2 = ax.twinx()
    ax2.plot(qs, is_feasible_all, label=f"feasible", color="#009E73", linestyle="dashed")
    ax2.set_ylabel(r"fraction feasible networks")
    ax2.set_ylim(-0.05, 1.05)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#009E73")
    ax2.yaxis.label.set_color("#009E73")
    ax2.tick_params(axis="y", colors="#009E73", which="both")
    ax.legend(loc="lower right")

    if nu == 1:
        ax.arrow(
            0, 2.5, 0, -1, length_includes_head=True, width=0.03, head_width=0.07,
            head_length=0.4
        )
    else:
        ax.arrow(
            0.2, 4, 0, -1, length_includes_head=True, width=0.03, head_width=0.07,
            head_length=0.4
        )

    if save_fig:
        plt.savefig(
            f"figures/ms/ms_adaptationandcongestion_nu{nu}.{format}",
            format=f"{format}"
        )

    print(f"Seed used: {seed}")


def ms_hysteresis_diff_q(
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

    fig.supxlabel(r"drivers of decline $d_A$")
    fig.supylabel(r"average pollinator abundance")
    plt.legend(
        [line_forward, line_backward], ["forward trajectory", "backward trajectory"],
        loc="center"
    )
    if save_fig:
        plt.savefig(f"figures/ms/hysteresis_pollinators_diff_q_mean_nu{nu}.{format}", format=f"{format}")

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

    fig.supxlabel(r"driver of decline $d_A$")
    fig.supylabel(r"pollinator abundance per species")\

    plt.legend(
        [line_forward, line_backward], ["forward trajectory", "backward trajectory"],
        loc="center"
    )
    if save_fig:
        plt.savefig(f"figures/ms/hysteresis_pollinators_diff_q_all_nu{nu}.{format}", format=f"{format}")


def ms_hysteresis_nu1(
    recalculate=False, format="png", fnumber=0, nu=1, q=0, dA_step=0.02,
    seed=None, ax=None
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

    # # mean abundance
    # fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    # ax1_plot, = ax.plot(dAs, A_forward_mean, color="#0072B2", label="Forward trajectory")
    # # ax.errorbar(
    # #     dAs, A_forward_mean, yerr=A_forward_std, capsize=5, fmt="none", color="#0072B2"
    # # )
    # ax1_fill = ax.fill_between(
    #     dAs, A_forward_mean-A_forward_std, A_forward_mean+A_forward_std,
    #     color="#0072B2", alpha=0.2
    # )
    # ax.plot(np.flip(dAs), A_backward_mean, color="#D55E00", label="Backward trajectory")
    # # ax.errorbar(
    # #     np.flip(dAs), A_backward_mean, yerr=A_backward_std, capsize=5, fmt="none",
    # #     color="#D55E00"
    # # )
    # ax.fill_between(
    #     np.flip(dAs), A_backward_mean-A_backward_std, A_backward_mean+A_backward_std,
    #     color="#D55E00", alpha=0.2
    # )
    # ax.set_xlabel(r"Drivers of decline $d_A$")
    # ax.set_ylabel(r"Average pollinator abundance")
    # dA_limit = min(dA_cutoff, dAs.max())
    # ax.set_xlim(0, dA_limit)
    # ax.set_ylim(-0.1, 2.2)
    # plt.legend()
    # if save_fig:
    #     plt.savefig(f"figures/ms/ms_hysteresis_pollinators_mean_nu{nu}.{format}", format=f"{format}")

    # abundances of each species

    collapse = dAs[np.argmax(A_sol_forward < 0.01, axis=0).max()]
    recovery = np.flip(dAs)[np.argmax(A_sol_backward > 0.01, axis=0).min()]
    print(collapse)
    print(recovery)

    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
        save_fig = True

    line_forward, *_ = ax.plot(
        dAs, A_sol_forward, color="#0072B2", linestyle="-"
    )
    line_backward, *_ = ax.plot(
        np.flip(dAs), A_sol_backward, color="#D55E00", linestyle="--"
    )
    ax.vlines(collapse, 0, 1.85, linestyles="dashed", colors="black")
    ax.vlines(recovery, 0, 1.85, linestyles="dashed", colors="black")
    ax.text(collapse, 1.85, "collapse", ha="center", va="bottom")
    ax.text(recovery, 1.85, "recovery", ha="center", va="bottom")

    ax.set_xlabel("driver of decline $d_A$\t\t\t", labelpad=-15)
    ax.set_ylabel("pollinator species\nequilibrium abundance")
    dA_limit = min(dA_cutoff, dAs.max())
    ax.set_xlim(0, dA_limit)
    ax.set_ylim(-0.1, 2.2)
    ax.set_xticks([collapse], [r"$d_A^{\mathrm{\ collapse}}$"])
    ax.set_yticks([])
    # plt.legend([line_forward, line_backward], ["Forward", "Backward"])
    if save_fig:
        plt.savefig(f"figures/ms/ms_hysteresis_pollinators_all_nu{nu}.{format}", format=f"{format}")


def ms_hysteresis_nu07(
    recalculate=False, format="png", fnumber=0, nu=1, q=0, dA_step=0.02,
    seed=None, ax=None
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
    dA_cutoff = 2

    # # mean abundance
    # fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    # ax1_plot, = ax.plot(dAs, A_forward_mean, color="#0072B2", label="Forward trajectory")
    # # ax.errorbar(
    # #     dAs, A_forward_mean, yerr=A_forward_std, capsize=5, fmt="none", color="#0072B2"
    # # )
    # ax1_fill = ax.fill_between(
    #     dAs, A_forward_mean-A_forward_std, A_forward_mean+A_forward_std,
    #     color="#0072B2", alpha=0.2
    # )
    # ax.plot(np.flip(dAs), A_backward_mean, color="#D55E00", label="Backward trajectory")
    # # ax.errorbar(
    # #     np.flip(dAs), A_backward_mean, yerr=A_backward_std, capsize=5, fmt="none",
    # #     color="#D55E00"
    # # )
    # ax.fill_between(
    #     np.flip(dAs), A_backward_mean-A_backward_std, A_backward_mean+A_backward_std,
    #     color="#D55E00", alpha=0.2
    # )
    # ax.set_xlabel(r"Drivers of decline $d_A$")
    # ax.set_ylabel(r"Average pollinator abundance")
    # dA_limit = min(dA_cutoff, dAs.max())
    # ax.set_xlim(0, dA_limit)
    # ax.set_ylim(-0.1, 2.2)
    # plt.legend()
    # if save_fig:
    #     plt.savefig(f"figures/ms/ms_hysteresis_pollinators_mean_nu{nu}.{format}", format=f"{format}")

    # abundances of each species

    collapse = dAs[np.argmax(A_sol_forward < 0.01, axis=0).max()]
    recovery = np.flip(dAs)[np.argmax(A_sol_backward > 0.01, axis=0)[1:].min()]
    print(np.argmax(A_sol_backward > 0.01, axis=0))
    print(collapse)
    print(recovery)

    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3), constrained_layout=True)
        save_fig = True

    line_forward, *_ = ax.plot(
        dAs, A_sol_forward, color="#0072B2", linestyle="-"
    )
    line_backward, *_ = ax.plot(
        np.flip(dAs), A_sol_backward, color="#D55E00", linestyle="--"
    )
    ax.vlines(collapse, 0, 3, linestyles="dashed", colors="black")
    ax.vlines(recovery, 0, 3, linestyles="dashed", colors="black")
    ax.text(collapse, 3, "collapse", ha="center", va="bottom")
    ax.text(recovery, 3, "recovery", ha="center", va="bottom")

    ax.set_xlabel(r"driver of decline $d_A$")
    ax.set_ylabel(r"pollinator abundance per species")
    dA_limit = min(dA_cutoff, dAs.max())
    ax.set_xlim(0, dA_limit)
    ax.set_ylim(-0.1, 3.5)
    # plt.legend([line_forward, line_backward], ["Forward", "Backward"])
    if save_fig:
        plt.savefig(f"figures/ms/ms_hysteresis_pollinators_all_nu{nu}.{format}", format=f"{format}")


def ms_rate_critical_dA_diff_q(
    recalculate=False, format="png", fnumber=0, seed=None, A_init=[0.1],
    qs=[0, 0.1, 0.2, 0.3, 0.4, 0.5], nu=1, legend=False, ax=None, xlabel=True
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

    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(3.5, 2.5), constrained_layout=True
        )
        save_fig = True
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
    if xlabel:
        ax.set_xlabel(r"rate of change $\lambda$")
    ax.set_ylabel(r"$d_A^{\mathrm{\ collapse}}$")
    if legend:
        ax.legend(loc="lower right", ncol=1, fontsize=10)

    if save_fig:
        plt.savefig(
            f"figures/ms/ms_rate_critical_dA_diff_q_nu{nu}_Ainit{A_init}.{format}",
            format=f"{format}"
        )

    print(f"Seed used: {seed}")


def ms_rate_dA_max(
    recalculate=False, format="png", seed=None, fnumber=0, q=0, nu=1, ax=None, fig=None
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

    thetas = np.linspace(0, 1, 41, endpoint=True)

    G = 1
    rate = 1

    n_reps = 100
    fname = f"output/rate_dA_max_G{G}_nu{nu}_q{q}_nreps{n_reps}_{fnumber}"

    if recalculate:

        S_init = 0.1

        A_final_all = np.zeros((n_reps, len(thetas), N_a))
        dA_collapse_all = np.zeros((n_reps))
        dA_maxs = np.zeros((n_reps, len(thetas)))
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
            y0_equilibrium = AM.equilibrium()
            A_final_base = y0_equilibrium[AM.N_p:AM.N]

            # find dA critical for equilibrium model
            y0 = np.full(AM.N, S_init, dtype=float)
            y0 = np.concatenate((y0, AM.alpha.flatten()))
            dA_collapse = AM.find_dA_collapse(dA_step=0.02, y0=y0)
            dA_collapse_all[rep] = dA_collapse

            AM.nu = nu
            AM.G = G
            AM.q = q
            t_end = int(1e5)
            n_steps = int(1e5)

            for j, theta in enumerate(thetas):
                dA_max = theta * dA_collapse
                dA_maxs[rep, j] = dA_max
                print(f"theta: {j+1} out of {len(thetas)}...")

                dA_dict = {
                    "func": util.dA_rate_max,
                    "args": (rate, dA_max)
                }

                y0 = np.full(AM.N, S_init, dtype=float)
                y0 = np.concatenate((y0, AM.alpha.flatten()))
                # y0 = np.concatenate((y0, np.array([0])))
                sol = AM.solve(
                    t_end, y0=y0, n_steps=n_steps, dA=dA_dict,
                    save_period=0, stop_on_equilibrium=True, stop_on_collapse=True
                )
                # print(util.dA_rate_max(sol.t[-1], rate, dA_max))

                A_final_all[rep, j] = AM.y[AM.N_p:AM.N, -1]

        np.savez(
            fname, thetas=thetas, dA_maxs=dA_maxs, A_final_all=A_final_all,
            A_final_base=A_final_base, dA_collapse_all=dA_collapse_all
        )

    try:
        with np.load(fname + ".npz") as sol:
            # print([sol1 for sol1 in sol])
            thetas = sol["thetas"]
            dA_maxs = sol["dA_maxs"]
            A_final_all= sol["A_final_all"]
            A_final_base = sol["A_final_base"]
    except FileNotFoundError:
        print(
            f"File not found. Make sure there exists an output file: "
            f"{fname}"
        )

    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(3.5, 3.5), constrained_layout=True, nrows=1, ncols=1
        )
        save_fig = True
    colors = ["#CC79A7", "#009E73", "#D55E00", "#E69F00", "#F0E442"]
    linestyles = ["--", "-.", ":", "--", "-."]
    y_max = np.max(A_final_all)
    ax_lines = []

    A_final = A_final_all[:, :, :]
    A_final_alive = np.sum(A_final > 0.01, axis=2, dtype=float)

    for rep in range(n_reps):
        with np.errstate(divide="ignore", invalid="ignore"):
            A_final_alive[rep] = np.nan_to_num(
                A_final_alive[rep] / A_final_alive[rep, 0], copy=False, nan=0
            )
        # remove any other (close to) inf values due to dividing by float 0.0
        for k in range(len(thetas)):
            if A_final_alive[rep, k] > 1e300:
                A_final_alive[rep, k] = 0

    A_final_alive_mean = np.mean(A_final_alive, axis=0)
    A_final_alive_std = np.std(A_final_alive, axis=0, ddof=1)

    A_final_alive_quartiles_1 = np.percentile(A_final_alive, 25, axis=0)

    A_final_alive_quartiles_3 = np.percentile(A_final_alive, 75, axis=0)

    ax_line, = ax.plot(
        thetas, A_final_alive_mean, linewidth=3.
    )
    ax_lines.append(ax_line)
    ax.fill_between(
        thetas, np.min((A_final_alive_quartiles_1, A_final_alive_mean), axis=0),
        A_final_alive_mean, alpha=0.35, color=ax_line.get_color(), linewidth=0
    )
    ax.fill_between(
        thetas, A_final_alive_mean,
        np.max((A_final_alive_mean, A_final_alive_quartiles_3), axis=0), alpha=0.35,
        color=ax_line.get_color(), linewidth=0
    )
    ax.set_xlabel(r"fraction $\theta$ of point of collapse $d_A^{\mathrm{\ collapse}}$")
    ax.set_ylabel("relative pollinator persistence")
    ax.set_xlim(thetas.min(), thetas.max())
    ax.set_ylim(-0.1, 1.1)

    if save_fig:
        plt.savefig(
            f"figures/ms/ms_rate_dA_max_G{G}_nu{nu}_q{q}_nreps{n_reps}_{fnumber}.{format}",
            format=f"{format}"
        )

    print(seed)


def ms_rate_dA_max_inset(
    recalculate=False, format="png", seed=None, fnumber=0, q=0, nu=1, ax=None, fig=None
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

    thetas = np.linspace(0, 1, 41, endpoint=True)

    G = 1
    rate = 1

    n_reps = 100
    fname = f"output/rate_dA_max_G{G}_nu{nu}_q{q}_nreps{n_reps}_{fnumber}"

    if recalculate:

        S_init = 0.1

        A_final_all = np.zeros((n_reps, len(thetas), N_a))
        dA_collapse_all = np.zeros((n_reps))
        dA_maxs = np.zeros((n_reps, len(thetas)))
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
            y0_equilibrium = AM.equilibrium()
            A_final_base = y0_equilibrium[AM.N_p:AM.N]

            # find dA critical for equilibrium model
            y0 = np.full(AM.N, S_init, dtype=float)
            y0 = np.concatenate((y0, AM.alpha.flatten()))
            dA_collapse = AM.find_dA_collapse(dA_step=0.02, y0=y0)
            dA_collapse_all[rep] = dA_collapse

            AM.nu = nu
            AM.G = G
            AM.q = q
            t_end = int(1e5)
            n_steps = int(1e5)

            for j, theta in enumerate(thetas):
                dA_max = theta * dA_collapse
                dA_maxs[rep, j] = dA_max
                print(f"theta: {j+1} out of {len(thetas)}...")

                dA_dict = {
                    "func": util.dA_rate_max,
                    "args": (rate, dA_max)
                }

                y0 = np.full(AM.N, S_init, dtype=float)
                y0 = np.concatenate((y0, AM.alpha.flatten()))
                # y0 = np.concatenate((y0, np.array([0])))
                sol = AM.solve(
                    t_end, y0=y0, n_steps=n_steps, dA=dA_dict,
                    save_period=0, stop_on_equilibrium=True, stop_on_collapse=True
                )
                # print(util.dA_rate_max(sol.t[-1], rate, dA_max))

                A_final_all[rep, j] = AM.y[AM.N_p:AM.N, -1]

        np.savez(
            fname, thetas=thetas, dA_maxs=dA_maxs, A_final_all=A_final_all,
            A_final_base=A_final_base, dA_collapse_all=dA_collapse_all
        )

    try:
        with np.load(fname + ".npz") as sol:
            # print([sol1 for sol1 in sol])
            thetas = sol["thetas"]
            dA_maxs = sol["dA_maxs"]
            A_final_all= sol["A_final_all"]
            A_final_base = sol["A_final_base"]
    except FileNotFoundError:
        print(
            f"File not found. Make sure there exists an output file: "
            f"{fname}"
        )

    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(3.5, 3.5), constrained_layout=True, nrows=1, ncols=1
        )
        save_fig = True
    colors = ["#CC79A7", "#009E73", "#D55E00", "#E69F00", "#F0E442"]
    linestyles = ["--", "-.", ":", "--", "-."]
    y_max = np.max(A_final_all)
    ax_lines = []

    A_final = A_final_all[:, :, :]
    A_final_alive = np.sum(A_final > 0.01, axis=2, dtype=float)

    for rep in range(n_reps):
        with np.errstate(divide="ignore", invalid="ignore"):
            A_final_alive[rep] = np.nan_to_num(
                A_final_alive[rep] / A_final_alive[rep, 0], copy=False, nan=0
            )
        # remove any other (close to) inf values due to dividing by float 0.0
        for k in range(len(thetas)):
            if A_final_alive[rep, k] > 1e300:
                A_final_alive[rep, k] = 0

    A_final_alive_mean = np.mean(A_final_alive, axis=0)
    A_final_alive_std = np.std(A_final_alive, axis=0, ddof=1)

    A_final_alive_quartiles_1 = np.percentile(A_final_alive, 25, axis=0)

    A_final_alive_quartiles_3 = np.percentile(A_final_alive, 75, axis=0)

    ax_line, = ax.plot(
        thetas, A_final_alive_mean, linewidth=3.
    )
    ax_lines.append(ax_line)
    ax.fill_between(
        thetas, np.min((A_final_alive_quartiles_1, A_final_alive_mean), axis=0),
        A_final_alive_mean, alpha=0.35, color=ax_line.get_color(), linewidth=0
    )
    ax.fill_between(
        thetas, A_final_alive_mean,
        np.max((A_final_alive_mean, A_final_alive_quartiles_3), axis=0), alpha=0.35,
        color=ax_line.get_color(), linewidth=0
    )
    ax.set_xlabel(r"fraction $\theta$ of point of collapse $d_A^{\mathrm{\ collapse}}$")
    ax.set_ylabel("relative pollinator persistence")
    ax.set_xlim(thetas.min(), thetas.max())
    ax.set_ylim(-0.1, 1.1)

    # inset
    # Create a set of inset Axes: these should fill the bounding box allocated to
    # them.
    ax2 = fig.add_axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    if nu == 1:
        ip = InsetPosition(ax, [0.1, 0.07, 0.62, 0.55])
    else:
        ip = InsetPosition(ax, [0.4, 0.4, 0.62, 0.55])

    ax2.set_axes_locator(ip)

    ts = np.linspace(0, 25, 100)

    rate_slow = 0.05
    rate_fast = 0.2

    low_max = 0.2
    med_max = 0.5
    high_max = 0.9
    full_max = 1

    # fig, ax2 = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

    fontsize = 9
    ax2.hlines(full_max, 0, ts.max(), linestyles="solid", colors="black")
    ax2.text(5, 1.05, r"$d_A^{\mathrm{\ collapse}}$ ($\theta=1$)", color="black", fontsize=fontsize)

    y = [util.dA_rate_max(t, rate_fast, high_max) for t in ts]
    ax2.plot(ts, y, linestyle="solid", color="#D55E00")
    ax2.text(18, 0.8, r"$\theta=0.9$", color="#D55E00", fontsize=fontsize)

    y = [util.dA_rate_max(t, rate_slow, high_max) for t in ts]
    ax2.plot(ts, y, linestyle="dashed", color="#D55E00")

    y = [util.dA_rate_max(t, rate_fast, med_max) for t in ts]
    ax2.plot(ts, y, linestyle="solid", color="#009E73")
    ax2.text(18, 0.4, r"$\theta=0.5$", color="#009E73", fontsize=fontsize)

    y = [util.dA_rate_max(t, rate_slow, med_max) for t in ts]
    ax2.plot(ts, y, linestyle="dashed", color="#009E73")

    y = [util.dA_rate_max(t, rate_fast, low_max) for t in ts]
    ax2.plot(ts, y, linestyle="solid", color="#CC79A7")
    ax2.text(18, 0.1, r"$\theta=0.2$", color="#CC79A7", fontsize=fontsize)

    y = [util.dA_rate_max(t, rate_slow, low_max) for t in ts]
    ax2.plot(ts, y, linestyle="dashed", color="#CC79A7")

    ax2.text(1.75, 0.25, "fast increase", color="black", rotation=76, fontsize=fontsize)
    ax2.text(6, 0.2, "slow increase", color="black", rotation=45, fontsize=fontsize)
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    ax2.set_xlabel("time", fontsize=fontsize)
    ax2.set_ylabel(r"driver of decline $d_A$", fontsize=fontsize)
    ax2.set_xlim(0, ts.max())
    ax2.set_ylim(0, full_max+0.2)
    ax2.spines.top.set_visible(True)
    ax2.spines.right.set_visible(True)

    if save_fig:
        plt.savefig(
            f"figures/ms/ms_rate_dA_max_G{G}_nu{nu}_q{q}_nreps{n_reps}_{fnumber}.{format}",
            format=f"{format}"
        )

    print(seed)


if __name__ == "__main__":

    #####################################################################################
    ### Figure 1C in manuscript
    #####################################################################################
    t0 = timer()
    ms_hysteresis_nu1(
        False, format="pdf", fnumber=2, nu=1, q=0
    )
    print(f"{timer()-t0:.2f} seconds")

    #####################################################################################
    ### Figure 2 in manuscript
    #####################################################################################
    t0 = timer()
    fig, (ax1, ax2) = plt.subplots(
        figsize=(7, 3.5), constrained_layout=True, nrows=1, ncols=2
    )
    format = "pdf"
    ms_abundance_rate_dependence_inset(
        False, format=format, fnumber=5, q=0, nu=1, ax=ax1, fig=fig
    )
    ms_rate_dA_max(
        False, format=format, fnumber=1, q=0, nu=1, ax=ax2
    )
    plt.savefig(
        f"figures/ms/ms_RIT_noadaptation.{format}",
        format=f"{format}"
    )
    print(f"{timer()-t0:.2f} seconds")

    #####################################################################################
    ### Figure 3 in manuscript
    #####################################################################################
    t0 = timer()
    fig, (ax1, ax2) = plt.subplots(
        figsize=(7, 3.5), constrained_layout=True, nrows=1, ncols=2
    )
    format = "pdf"
    ms_abundance_rate_dependence(
        False, format=format, fnumber=5, q=0.2, nu=0.7, ax=ax1
    )
    ms_rate_dA_max_inset(
        False, format=format, fnumber=1, q=0.2, nu=0.7, ax=ax2, fig=fig
    )
    plt.savefig(
        f"figures/ms/ms_RIT_adaptation.{format}",
        format=f"{format}"
    )
    print(f"{timer()-t0:.2f} seconds")

    #####################################################################################
    ### Figure 4 in manuscript
    #####################################################################################
    t0 = timer()
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(7, 2.5), constrained_layout=True
    )
    ms_bifurcation_feasibility_diff_q(
        False, format="pdf", fnumber=0, seed=None, nu=1, ax=ax1
    )
    ms_bifurcation_feasibility_diff_q(
        False, format="pdf", fnumber=2, seed=None, nu=0.7, ax=ax2
    )
    format = "pdf"
    plt.savefig(
        f"figures/ms/ms_adaptationandcongestion.{format}",
        format=f"{format}"
    )
    print(f"{timer()-t0:.2f} seconds")

    #####################################################################################
    ### Figure S2-5 in SI
    #####################################################################################
    t0 = timer()
    ms_hysteresis_diff_q(
        False, True, format="pdf", fnumber=2, nu=1,
        dA_step=0.02
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    ms_hysteresis_diff_q(
        False, True, format="pdf", fnumber=2, nu=0.8,
        dA_step=0.02
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    ms_hysteresis_diff_q(
        False, True, format="pdf", fnumber=2, nu=0.7,
        dA_step=0.02
    )
    print(f"{timer()-t0:.2f} seconds")

    t0 = timer()
    ms_hysteresis_diff_q(
        False, True, format="pdf", fnumber=2, nu=0.6,
        dA_step=0.02
    )
    print(f"{timer()-t0:.2f} seconds")

    #####################################################################################
    ### Figure S6 in SI
    #####################################################################################
    t0 = timer()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2, ncols=2, figsize=(6.5, 5.5), constrained_layout=True
    )
    ms_rate_critical_dA_diff_q(
        False, fnumber=0, format=format, qs=[0, 0.2, 0.4], nu=1,
        A_init=[0.1], ax=ax1, xlabel=False
    )
    ms_rate_critical_dA_diff_q(
        False, fnumber=1, format=format, qs=[0, 0.2, 0.4], nu=1,
        A_init=[2], ax=ax2, legend=True, xlabel=False
    )
    ms_rate_critical_dA_diff_q(
        False, fnumber=0, format=format, qs=[0, 0.2, 0.4], nu=0.7,
        A_init=[0.1], ax=ax3
    )
    ms_rate_critical_dA_diff_q(
        False, fnumber=1, format=format, qs=[0, 0.2, 0.4], nu=0.7,
        A_init=[2], ax=ax4
    )
    format = "pdf"
    plt.savefig(
        f"figures/ms/ms_rate_critical_dA_diff_q.{format}",
        format=f"{format}"
    )
    print(f"{timer()-t0:.2f} seconds")

    plt.show()
