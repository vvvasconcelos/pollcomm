#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 24/01/2022
# ---------------------------------------------------------------------------
""" visualization.py

Functions used to plot data
"""
# ---------------------------------------------------------------------------
import copy

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np

import pollcomm as pc

from cycler import cycler
line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))
marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))

plt.rc("axes", prop_cycle=line_cycler)
# plt.rc("axes", prop_cycle=marker_cycler)
plt.rc("font", family="serif", size=18.)
plt.rc("savefig", dpi=200)
plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
plt.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)


def anim_alpha(sol, pollcomm, dA=None):
    fig, ax = plt.subplots()
    ax.set_title(f"Alpha, dA = {dA}")

    # sort gamma from most generalist to most specialist species)
    for i in range(sol.y_partial.shape[1]):
        alpha = sol.y_partial[:, i].reshape((pollcomm.N_p, pollcomm.N_a))
        alpha = pc.sort_network(alpha)
        ax.imshow(alpha, cmap='Greys', interpolation='nearest')
        ax.set_title(f"Alpha: {i}/{sol.y_partial.shape[1]}")
        ax.set_xlabel("pollinators")
        ax.set_ylabel("plants")
        plt.draw()
        plt.pause(0.0001)
        plt.cla()

    return fig, ax


def plot_alpha_beta_network(
    network, alpha, beta, forbidden_network, save_fig=False, title=None
):

    # sort all matrices from generalists to specialists according to alpha
    alpha_sort = copy.deepcopy(alpha)
    alpha_sort[alpha_sort < 0.01] = 0
    alpha_sort[alpha_sort >= 0.01] = 1
    alpha_sort, alpha, network, beta, forbidden_network = pc.sort_network(
        alpha_sort, alpha, network, beta, forbidden_network
    )
    # network, alpha, beta, forbidden_network = pc.sort_network(
    #     network, alpha, beta, forbidden_network
    # )
    # sort network on its own
    # network = pc.sort_network(network)


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, constrained_layout=False, sharex=True, sharey=True
    )
    if title is None:
        fig.suptitle(f"Network matrices AM")
    else:
        fig.suptitle(title)
    cmap = "viridis"

    ax1.matshow(network, cmap=cmap, origin="lower")
    ax1.set_title(f"Network")
    ax1.set_ylabel("Plants")

    ax2.matshow(forbidden_network, cmap=cmap, origin="lower")
    ax2.set_title("Forbidden links")

    ax3.matshow(alpha, cmap=cmap, origin="lower")
    ax3.set_title(r"Foraging $\alpha$")
    ax3.set_xlabel("Pollinators")
    ax3.set_ylabel("Plants")
    ax3.xaxis.set_ticks_position('bottom')

    im = ax4.matshow(beta, cmap=cmap, origin="lower")
    ax4.set_title(r"Trait $\beta$")
    ax4.set_xlabel("Pollinators")
    ax4.xaxis.set_ticks_position('bottom')

    fig.subplots_adjust(bottom=0.15, top=0.85, left=0.1, right=0.85,
                    wspace=0.01, hspace=0.3)

    # add axes for colorbar at (0.83, 0.1) with axes width and height (0.02, 0.8)
    cax = fig.add_axes([0.85, 0.1, 0.04, 0.75])
    cbar = fig.colorbar(im, cax=cax)

    if save_fig:
        plt.savefig(f"figures/network_all.png", format="png", dpi=500)

    return


def plot_AM_alpha_init_alpha_end_beta(
    alpha_init, alpha_end, beta_P, beta_A, save_fig=False, title=None, cmap="viridis"
):

    # sort all matrices from generalists to specialists
    alpha_init_sort = copy.deepcopy(alpha_init)
    alpha_init_sort[alpha_init_sort < 0.5/alpha_init.shape[0]] = 0
    alpha_init_sort[alpha_init_sort >= 0.5/alpha_init.shape[0]] = 1
    alpha_init_sort, alpha_init, alpha_end, beta_P, beta_A = pc.sort_network(
        alpha_init_sort, alpha_init, alpha_end, beta_P, beta_A
    )

    # alpha_end_sort = copy.deepcopy(alpha_end)
    # alpha_end_sort[alpha_end_sort < 0.5/alpha_init.shape[0]] = 0
    # alpha_end_sort[alpha_end_sort >= 0.5/alpha_init.shape[0]] = 1
    # alpha_end_sort, alpha_end = pc.sort_network(alpha_end_sort, alpha_end)
    #
    # beta_sort = copy.deepcopy(beta)
    # beta_sort[beta_sort == 0] = 0
    # beta_sort[beta_sort > 0] = 1
    # beta_sort, beta = pc.sort_network(beta_sort, beta)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, constrained_layout=True, sharex=True, sharey=True
    )
    # if title is None:
    #     fig.suptitle("Adaptive Model", y=0.8)
    # else:
    #     fig.suptitle(title, y=0.8)

    im1 = ax1.matshow(beta_P, cmap=cmap, origin="lower")
    ax1.set_title(r"$\beta_P$")
    ax1.set_ylabel("Plants")
    # ax1.xaxis.set_ticks_position('bottom')
    # ax1.set_xlabel("Pollinators")
    plt.colorbar(im1, ax=ax1, fraction=0.05)

    im2 = ax2.matshow(beta_A, cmap=cmap, origin="lower")
    ax2.set_title(r"$\beta_A$")
    ax2.set_ylabel("Plants")
    # ax2.xaxis.set_ticks_position('bottom')
    # ax2.set_xlabel("Pollinators")
    plt.colorbar(im2, ax=ax2, fraction=0.05)

    im3 = ax3.matshow(alpha_init, cmap=cmap, origin="lower")
    ax3.set_title(r"$\alpha_{\mathrm{init}}$")
    ax3.xaxis.set_ticks_position('bottom')
    ax3.set_xlabel("Pollinators")
    ax3.set_ylabel("Plants")
    plt.colorbar(im3, ax=ax3, fraction=0.05)

    im4 = ax4.matshow(alpha_end, cmap=cmap, origin="lower")
    ax4.set_title(r"$\alpha_{\mathrm{end}}$")
    ax4.xaxis.set_ticks_position('bottom')
    ax4.set_xlabel("Pollinators")
    plt.colorbar(im4, ax=ax4, fraction=0.05)

    # fig.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9,
                    # wspace=0.2, hspace=None)

    if save_fig:
        plt.savefig(f"figures/AM_alpha_init_alpha_end_beta_nested_norm.png", format="png", dpi=500)

    return


def plot_VM_alpha_init_alpha_end(
    alpha_init, alpha_end, save_fig=False, title=None, cmap="viridis"
):

    # sort all matrices from generalists to specialists
    alpha_init_sort = copy.deepcopy(alpha_init)
    alpha_init_sort[alpha_init_sort < 0.01] = 0
    alpha_init_sort[alpha_init_sort >= 0.01] = 1
    alpha_init_sort, alpha_init, alpha_end = pc.sort_network(
        alpha_init_sort, alpha_init, alpha_end
    )

    # alpha_end_sort = copy.deepcopy(alpha_end)
    # alpha_end_sort[alpha_end_sort < 0.01] = 0
    # alpha_end_sort[alpha_end_sort >= 0.01] = 1
    # alpha_end_sort, alpha_end = pc.sort_network(alpha_end_sort, alpha_end)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, constrained_layout=True, sharex=True, sharey=True
    )
    if title is None:
        fig.suptitle("Valdovinos Model", y=0.8)
    else:
        fig.suptitle(title, y=0.8)

    im1 = ax1.matshow(alpha_init, cmap=cmap, origin="lower")
    ax1.set_title(r"$\alpha_{\mathrm{init}}$")
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xlabel("Pollinators")
    plt.colorbar(im1, ax=ax1, fraction=0.05)

    im2 = ax2.matshow(alpha_end, cmap=cmap, origin="lower")
    ax2.set_title(r"$\alpha_{\mathrm{end}}$")
    ax2.xaxis.set_ticks_position('bottom')
    ax2.set_xlabel("Pollinators")
    plt.colorbar(im2, ax=ax2, fraction=0.05)

    # fig.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9,
                    # wspace=0.2, hspace=None)

    if save_fig:
        plt.savefig(f"figures/VM_alpha_init_alpha_end.png", format="png", dpi=500)

    return


def plot_state_space_AM_BM(fname_AM, fname_BM):

    with np.load(fname_AM) as sol:
        dAs = sol["dAs"]
        P_mean = sol["P_mean"]
        A_mean = sol["A_mean"]
        P_backward_mean = sol["P_backward_mean"]
        A_backward_mean = sol["A_backward_mean"]
    with np.load(fname_BM) as sol_BM:
        dAs_BM = sol_BM["dAs_BM"]
        P_mean_BM = sol_BM["P_mean_BM"]
        A_mean_BM = sol_BM["A_mean_BM"]
        P_backward_mean_BM = sol_BM["P_backward_mean_BM"]
        A_backward_mean_BM = sol_BM["A_backward_mean_BM"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, tight_layout=True, sharex=True, sharey=True
    )
    fig.suptitle("Steady-state abundance as function \nof drivers of decline $d_A$")
    ax1.set_title("Base Model")
    ax1.set_ylabel(r"$\langle P \rangle$")
    forward = ax1.scatter(dAs_BM, P_mean_BM, color="blue")
    backward = ax1.scatter(np.flip(dAs_BM), P_backward_mean_BM, color="red")
    ax1.legend(
        [forward, backward],
        [
            r"Increasing $d_A$",
            r"Decreasing $d_A$"
        ], loc="upper right"
    )

    ax2.set_title("Adaptive Model")
    ax2.scatter(dAs, P_mean, color="blue")
    ax2.scatter(np.flip(dAs), P_backward_mean, color="red")

    ax3.set_xlabel(r"Driver of decline $d_A$")
    ax3.set_ylabel(r"$\langle A \rangle$")
    ax3.scatter(dAs_BM, A_mean_BM, color="blue")
    ax3.scatter(np.flip(dAs_BM), A_backward_mean_BM, color="red")

    ax4.set_xlabel(r"Driver of decline $d_A$")
    ax4.scatter(dAs, A_mean, color="blue")
    ax4.scatter(np.flip(dAs), A_backward_mean, color="red")


    plt.savefig(f"figures/state_space_AM_BM_nu_low.png", format="png", dpi=500)


def plot_state_space_AM_BM_VM(fname_AM, fname_BM, fname_VM):

    with np.load(fname_BM) as sol_BM:
        dAs_BM = sol_BM["dAs_BM"]
        P_mean_BM = sol_BM["P_mean_BM"]
        A_mean_BM = sol_BM["A_mean_BM"]
        P_backward_mean_BM = sol_BM["P_backward_mean_BM"]
        A_backward_mean_BM = sol_BM["A_backward_mean_BM"]
    with np.load(fname_AM) as sol_AM:
        dAs_AM = sol_AM["dAs"]
        P_mean_AM = sol_AM["P_mean"]
        A_mean_AM = sol_AM["A_mean"]
        P_backward_mean_AM = sol_AM["P_backward_mean"]
        A_backward_mean_AM = sol_AM["A_backward_mean"]
    with np.load(fname_VM) as sol_VM:
        dAs_VM = sol_VM["dAs"]
        P_mean_VM = sol_VM["P_mean"]
        A_mean_VM = sol_VM["A_mean"]
        P_backward_mean_VM = sol_VM["P_backward_mean"]
        A_backward_mean_VM = sol_VM["A_backward_mean"]

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
        2, 3, tight_layout=True, sharex="col", sharey=False
    )

    # some figure variables
    s = 10
    fig.suptitle(
        "Steady-state abundance as function of drivers of decline $d_A$\n" +
        "Blue lines: increasing $d_A$\n" +
        "Red lines: decreasing $d_A$"
    )

    ax1.set_title("Base Model")
    ax1.set_ylabel(r"$\langle P \rangle$")
    forward, = ax1.plot(dAs_BM, P_mean_BM, color="blue")
    backward, = ax1.plot(np.flip(dAs_BM), P_backward_mean_BM, color="red")
    # ax1.legend(
    #     [forward, backward],
    #     [
    #         r"Increasing $d_A$",
    #         r"Decreasing $d_A$"
    #     ], loc="upper right"
    # )

    ax2.set_title("Adaptive Model")
    ax2.set_ylabel(r"$\langle P \rangle$")
    ax2.plot(dAs_AM, P_mean_AM, color="blue")
    ax2.plot(np.flip(dAs_AM), P_backward_mean_AM, color="red")

    ax3.set_title("Valdovinos Model")
    ax3.set_ylabel(r"$\langle P \rangle$")
    ax3.plot(dAs_VM, P_mean_VM, color="blue")
    ax3.plot(np.flip(dAs_VM), P_backward_mean_VM, color="red")

    ax4.set_xlabel(r"Driver of decline $d_A$")
    ax4.set_ylabel(r"$\langle A \rangle$")
    ax4.plot(dAs_BM, A_mean_BM, color="blue")
    ax4.plot(np.flip(dAs_BM), A_backward_mean_BM, color="red")

    ax5.set_xlabel(r"Driver of decline $d_A$")
    ax5.set_ylabel(r"$\langle A \rangle$")
    ax5.plot(dAs_AM, A_mean_AM, color="blue")
    ax5.plot(np.flip(dAs_AM), A_backward_mean_AM, color="red")

    ax6.set_xlabel(r"Driver of decline $d_A$")
    ax6.set_ylabel(r"$\langle A \rangle$")
    ax6.plot(dAs_VM, A_mean_VM, color="blue")
    ax6.plot(np.flip(dAs_VM), A_backward_mean_VM, color="red")

    plt.savefig(f"figures/state_space_AM_BM_VM2.png", format="png", dpi=500)


def plot_state_space_rate_AM_BM(fname_AM, fname_BM):

    with np.load(fname_AM) as sol:
        dA_maxs = sol["dA_maxs"]
        rates = sol["rates"]
        P_means = sol["P_means"]
        A_means = sol["A_means"]

    with np.load(fname_BM) as sol_BM:
        dA_maxs_BM = sol_BM["dA_maxs"]
        rates_BM = sol_BM["rates"]
        P_means_BM = sol_BM["P_means"]
        A_means_BM = sol_BM["A_means"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, tight_layout=False, sharex=True, sharey=True
    )
    fig.suptitle(
        "Steady-state abundance as function of \nrate of change of drivers of decline $d_A$"
    )
    ax1.set_title("No Adaptive Foraging")
    ax1.set_ylabel(r"$\langle P \rangle$")
    for i, P_mean_BM in enumerate(P_means_BM):
        ax1.plot(rates_BM, P_mean_BM)

    ax2.set_title("Adaptive Foraging")
    for i, P_mean in enumerate(P_means):
        ax2.plot(rates, P_mean, label=f"$d_{{A_{{max}}}}$ = {dA_maxs_BM[i]}")
    ax2.legend(bbox_to_anchor=(1.03, 1.2))

    # ax3.set_title("No AF")
    ax3.set_xlabel(r"Rate of change of $d_A$")
    ax3.set_ylabel(r"$\langle A \rangle$")
    for i, A_mean_BM in enumerate(A_means_BM):
        ax3.plot(rates_BM, A_mean_BM)

    # ax4.set_title("AF")
    ax4.set_xlabel(r"Rate of change of $d_A$")
    for i, A_mean in enumerate(A_means):
        ax4.plot(rates, A_mean)

    plt.subplots_adjust(left=0.1, bottom=0.156, right=0.71, top=0.77, wspace=0.3, hspace=0.18)

    plt.savefig(f"figures/state_space_rate_AM_BM.png", format="png", dpi=500)


def plot_state_space_rate_AM_BM_VM(fname_AM, fname_BM, fname_VM):

    with np.load(fname_AM) as sol_AM:
        dA_maxs_AM = sol_AM["dA_maxs"]
        rates_AM = sol_AM["rates"]
        P_means_AM = sol_AM["P_means"]
        A_means_AM = sol_AM["A_means"]

    with np.load(fname_BM) as sol_BM:
        dA_maxs_BM = sol_BM["dA_maxs"]
        rates_BM = sol_BM["rates"]
        P_means_BM = sol_BM["P_means"]
        A_means_BM = sol_BM["A_means"]

    with np.load(fname_VM) as sol_VM:
        dA_maxs_VM = sol_VM["dA_maxs"]
        rates_VM = sol_VM["rates"]
        P_means_VM = sol_VM["P_means"]
        A_means_VM = sol_VM["A_means"]

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
        2, 3, tight_layout=True, sharex="col", sharey=False
    )
    fig.suptitle(
        "Final abundance as function of rate of change of drivers of decline $d_A$\n" +
        "(simulation stopped after reaching $d_{A_{max}}$, so no steady-state)"
    )
    # fig.suptitle(
    #     "Steady-state abundance as function of rate of change of drivers of decline $d_A$\n" +
    #     "(simulation continued after reaching $d_{A_{max}}$ until steady-state reached)"
    # )

    ax1.set_title("Base Model")
    ax1.set_ylabel(r"$\langle P \rangle$")
    for i, P_mean_BM in enumerate(P_means_BM):
        ax1.plot(rates_BM, P_mean_BM)

    ax2.set_title("Adaptive Model")
    ax2.set_ylabel(r"$\langle P \rangle$")
    for i, P_mean_AM in enumerate(P_means_AM):
        ax2.plot(rates_AM, P_mean_AM, label=f"$d_{{A_{{max}}}}$ = {dA_maxs_AM[i]}")
    # ax2.legend(bbox_to_anchor=(1.03, 1.2))

    ax3.set_title("Valdovinos Model")
    ax3.set_ylabel(r"$\langle P \rangle$")
    for i, P_mean_VM in enumerate(P_means_VM):
        ax3.plot(rates_VM, P_mean_VM)

    ax4.set_xlabel(r"Rate of change of $d_A$")
    ax4.set_ylabel(r"$\langle A \rangle$")
    for i, A_mean_BM in enumerate(A_means_BM):
        ax4.plot(rates_BM, A_mean_BM)

    ax5.set_xlabel(r"Rate of change of $d_A$")
    ax5.set_ylabel(r"$\langle A \rangle$")
    for i, A_mean_AM in enumerate(A_means_AM):
        ax5.plot(rates_AM, A_mean_AM)

    ax6.set_xlabel(r"Rate of change of $d_A$")
    ax6.set_ylabel(r"$\langle A \rangle$")
    for i, A_mean_VM in enumerate(A_means_VM):
        ax6.plot(rates_VM, A_mean_VM)

    # plt.subplots_adjust(left=0.1, bottom=0.156, right=0.71, top=0.77, wspace=0.3, hspace=0.18)

    # plt.savefig(f"figures/state_space_rate_AM_BM_VM.png", format="png", dpi=500)
    plt.savefig(f"figures/state_space_rate_AM_BM_VM_no_const.png", format="png", dpi=500)


def plot_state_space_rate_ARM(fname_ARM, save_fig=False):

    with np.load(fname_ARM) as sol:
        dA_maxs = sol["dA_maxs"]
        rates = sol["rates"]
        P_means = sol["P_means"]
        A_means = sol["A_means"]
        R_means = sol["R_means"]

    fig, axs = plt.subplots(constrained_layout=True)
    fig.suptitle("ARM Phase space plants")
    for i, P_mean in enumerate(P_means):
        axs.plot(rates, P_mean, label=f"dA_max = {dA_maxs[i]}")
    axs.set_xlabel(r"Rate")
    axs.set_ylabel(r"$\langle P \rangle$")
    # axs.set_ylim(0, 2)
    axs.legend()
    if save_fig:
        plt.savefig(f"figures/state_space_rate_ARM.png", format="png", dpi=300)

    fig, axs = plt.subplots(constrained_layout=True)
    fig.suptitle("ARM Phase space pollinators")
    for i, A_mean in enumerate(A_means):
        axs.plot(rates, A_mean, label=f"dA_max = {dA_maxs[i]}")
    axs.set_xlabel(r"Rate")
    axs.set_ylabel(r"$\langle A \rangle$")
    # axs.set_ylim(0, 2)
    axs.legend()
    if save_fig:
        plt.savefig(f"figures/state_space_rate_ARM.png", format="png", dpi=300)

    fig, axs = plt.subplots(constrained_layout=True)
    fig.suptitle("ARM Phase space plants")
    for i, R_mean in enumerate(R_means):
        axs.plot(rates, R_mean, label=f"dA_max = {dA_maxs[i]}")
    axs.set_xlabel(r"Rate")
    axs.set_ylabel(r"$\langle R \rangle$")
    # axs.set_ylim(0, 2)
    axs.legend()
    if save_fig:
        plt.savefig(f"figures/state_space_rate_ARM.png", format="png", dpi=300)


def plot_time_sol_pollcomm(sol, pollcomm, dA=None, save_fig=False):

    fig, (ax1, ax2) = plt.subplots(ncols=2, constrained_layout=True)
    for i in range(pollcomm.N_p):
        line_plants, = ax1.plot(sol.t, sol.y[i])
    for i in range(pollcomm.N_p, pollcomm.N):
        line_polls, = ax2.plot(sol.t, sol.y[i])

    if dA is not None:
        plt.title(f"$d_A = {dA}$")
    ax1.set_title("Plants")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Abundance")

    ax2.set_title("Pollinators")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Abundance")
    # plt.legend([line_polls], ["Pollinators"], loc="upper right")
    # plt.legend([line_plants, line_polls], ["Plants", "Pollinators"], loc="upper right")

    if save_fig:
        if dA is None:
            plt.savefig(f"figures/time_sol_pollcomm.svg", format="svg", dpi=300)
        else:
            plt.savefig(f"figures/time_sol_pollcomm_dA{dA}.svg", format="svg", dpi=300)


def plot_time_sol_ARM(sol, ARM, t_end=None, t_remove=None, dA=None, save_fig=False):

    if isinstance(dA, (int, float)):
        pass
    elif isinstance(dA, (dict)):
        dAs = []
        for i, t in enumerate(sol.t):
            dA_val = dA["func"](t, *dA.get("args", None))
            dAs.append(dA_val)

    if t_remove is not None:
        ind_remove = int(t_remove * sol.t.shape[0])
        sol.t = sol.t[ind_remove:]
        sol.y = sol.y[:, ind_remove:]

        if dAs:
            dAs = dAs[ind_remove:]
            fig, ax = plt.subplots()
            fig.suptitle(f"$d_A$ as a function of time. $r={dA['args'][0]}$")
            ax.plot(sol.t, dAs, color="black", linestyle="dashed")
            ax.set_xlabel("Time")
            ax.set_ylabel(r"$d_A$")
            plt.savefig(f"figures/time_sol_ARM_dAs.png", format="png", dpi=500)

    # plot pollinators
    fig, ax1 = plt.subplots()
    for i in range(ARM.N_p, ARM.N):
        line_polls, = ax1.plot(sol.t, sol.y[i])

    plt.title(f"Pollinators")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Abundance of pollinators")
    plt.tight_layout()
    plt.savefig(f"figures/time_sol_ARM_polls.png", format="png", dpi=500)


    # if save_fig:
    #     if dA is None:
    #         plt.savefig(f"figures/time_sol_ARM.svg", format="svg", dpi=300)
    #     else:
    #         plt.savefig(f"figures/time_sol_ARM_dA{dA}.svg", format="svg", dpi=300)

    # plot plants
    fig, ax1 = plt.subplots()
    for i in range(ARM.N_p):
        line_plants, = ax1.plot(sol.t, sol.y[i])

    plt.title(f"Plants")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Abundance of plants")
    plt.tight_layout()
    plt.savefig(f"figures/time_sol_ARM_plants.png", format="png", dpi=500)


    # plot resources
    fig, ax1 = plt.subplots()
    for i in range(ARM.N, ARM.N + ARM.N_p):
        line_resources, = ax1.plot(sol.t, sol.y[i])

    plt.title(f"Resources")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Abundance of resources")
    plt.tight_layout()
    plt.savefig(f"figures/time_sol_ARM_resources.png", format="png", dpi=500)


def plot_time_sol_AM(sol, ARM, t_end=None, t_remove=None, dA=None, save_fig=False):

    dAs = []
    if isinstance(dA, (int, float)):
        pass
    elif isinstance(dA, (dict)):
        for i, t in enumerate(sol.t):
            dA_val = dA["func"](t, *dA.get("args", None))
            dAs.append(dA_val)

    if t_remove is not None:
        ind_remove = int(t_remove * sol.t.shape[0])
        sol.t = sol.t[ind_remove:]
        sol.y = sol.y[:, ind_remove:]

        if dAs:
            dAs = dAs[ind_remove:]
    if dAs:
        fig, ax = plt.subplots()
        fig.suptitle(f"$d_A$ as a function of time. $r={dA['args'][0]}$")
        ax.plot(sol.t, dAs, color="black", linestyle="dashed")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$d_A$")
        plt.savefig(f"figures/time_sol_AM_dAs.png", format="png", dpi=500)

    # plot pollinators
    fig, ax1 = plt.subplots()
    for i in range(ARM.N_p, ARM.N):
        line_polls, = ax1.plot(sol.t, sol.y[i])

    plt.title(f"Pollinators")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Abundance of pollinators")
    plt.tight_layout()
    plt.savefig(f"figures/time_sol_AM_polls.png", format="png", dpi=500)

    # if save_fig:
    #     if dA is None:
    #         plt.savefig(f"figures/time_sol_ARM.svg", format="svg", dpi=300)
    #     else:
    #         plt.savefig(f"figures/time_sol_ARM_dA{dA}.svg", format="svg", dpi=300)

    # plot plants
    fig, ax1 = plt.subplots()
    for i in range(ARM.N_p):
        line_plants, = ax1.plot(sol.t, sol.y[i])

    plt.title(f"Plants")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Abundance of plants")
    plt.tight_layout()
    plt.savefig(f"figures/time_sol_AM_plants.png", format="png", dpi=500)


def plot_degree_abundance(model, dA=None, save_fig=False):

    alpha_end = model.y_partial[:, -1].reshape((model.N_p, model.N_a))
    # alpha_init = model.alpha
    # beta_P, beta_A = model.beta_P, model.beta_A

    alpha_end_degree = copy.deepcopy(alpha_end)
    alpha_end_degree[alpha_end_degree < 0.5/alpha_end.shape[0]] = 0
    alpha_end_degree[alpha_end_degree >= 0.5/alpha_end.shape[0]] = 1

    plant_degrees = alpha_end_degree.sum(axis=1)
    poll_degrees = alpha_end_degree.sum(axis=0)

    fig, ax1 = plt.subplots(constrained_layout=True)
    # print(model.y[:model.N_p, :])
    for i in range(model.N_p):
        line_plants = ax1.scatter(
            [plant_degrees[i]], [model.y[i, -1]], color="green"
        )
    for i in range(model.N_p, model.N):
        line_polls = ax1.scatter(
            [poll_degrees[i-model.N_p]], [model.y[i, -1]], color="blue",
        )
    plt.title(f"Abundance as function of number of interactions")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Abundance")
    ax1.legend([line_plants, line_polls], ["Plant", "Pollinator"])
    if save_fig:
        plt.savefig(f"figures/plot_degree_abundance.png", format="png", dpi=500)
    #
    #
    # fig, ax1 = plt.subplots(constrained_layout=True)
    # for i in range(model.N_p, model.N):
    # plt.title(f"Pollinators")
    # ax1.set_xlabel("Degree")
    # ax1.set_ylabel("Abundance of pollinators")
    # if save_fig:
    #     plt.savefig(f"figures/plot_degree_abundance_polls.png", format="png", dpi=500)


def plot_time_sol_VM(sol, VM, dA=None, save_fig=False):

    if isinstance(dA, (int, float)):
        pass
    elif isinstance(dA, (dict)):
        dAs = []
        for i, t in enumerate(sol.t):
            dA_val = dA["func"](t, *dA.get("args", None))
            dAs.append(dA_val)

        fig, ax = plt.subplots()
        fig.suptitle(f"$d_A$ as a function of time. $r={dA['args'][0]}$")
        ax.plot(sol.t, dAs, color="black", linestyle="dashed")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$d_A$")
        plt.savefig(f"figures/time_sol_ARM_dAs.png", format="png", dpi=500)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, tight_layout=True, sharex=False, sharey=False
    )
    fig.suptitle(r"Time series Valdovinos model and final foraging effort $\alpha$")
    ax1.set_ylabel("Plants")
    ax1.set_xlabel("Time")
    for i in range(VM.N_p):
        line_plants, = ax1.plot(sol.t, sol.y[i])

    ax2.set_ylabel("Pollinators")
    ax2.set_xlabel("Time")
    for i in range(VM.N_p, VM.N):
        line_polls, = ax2.plot(sol.t, sol.y[i])

    ax3.set_ylabel("Resources")
    ax3.set_xlabel("Time")
    for i in range(VM.N, VM.N + VM.N_p):
        line_resources, = ax3.plot(sol.t, sol.y[i])

    alpha = copy.deepcopy(sol.y_partial[:, -1].reshape((VM.N_p, VM.N_a)))
    network = copy.deepcopy(alpha)
    network[network < 0.01] = 0
    network[network >= 0.01] = 1
    network, alpha = pc.sort_network(network, alpha)

    ax4.set_xlabel("Pollinators")
    ax4.set_ylabel("Plants")
    cmap = "viridis"
    ax4.matshow(alpha, cmap=cmap, origin="lower")
    ax4.xaxis.set_ticks_position('bottom')

    # plt.subplots_adjust(left=0.1, bottom=0.156, right=0.71, top=0.77, wspace=0.3, hspace=0.18)
    if save_fig:
        plt.savefig(f"figures/time_sol_VM.png", format="png", dpi=500)


def plot_time_sol_AM_BM_VM(AM_sol, AM, BM_sol, BM, VM_sol, VM, dA=None, save_fig=False):

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
        2, 3, tight_layout=True, sharex="col", sharey=False
    )

    if dA is not None:
        fig.suptitle(
            f"Time series $d_A={dA}$"
        )
    else:
        fig.suptitle(
            f"Time series"
        )

    ax1.set_title("Base Model")
    ax1.set_ylabel(r"$\langle P \rangle$")
    ax1.set_ylim(0, 1.05*BM_sol.y[:BM.N_p].max())
    for i in range(BM.N_p):
        ax1.plot(BM_sol.t, BM_sol.y[i])

    ax2.set_title("Adaptive Model")
    ax2.set_ylabel(r"$\langle P \rangle$")
    ax2.set_ylim(0, 1.05*AM_sol.y[:AM.N_p].max())
    for i in range(AM.N_p):
        ax2.plot(AM_sol.t, AM_sol.y[i])

    ax3.set_title("Valdovinos Model")
    ax3.set_ylabel(r"$\langle P \rangle$")
    ax3.set_ylim(0, 1.05*VM_sol.y[:VM.N_p].max())
    for i in range(VM.N_p):
        ax3.plot(VM_sol.t, VM_sol.y[i])

    ax4.set_xlabel("Time")
    ax4.set_ylabel(r"$\langle A \rangle$")
    ax4.set_ylim(0, 1.05*BM_sol.y[BM.N_p:BM.N].max())
    for i in range(BM.N_p, BM.N):
        ax4.plot(BM_sol.t, BM_sol.y[i])

    ax5.set_xlabel("Time")
    ax5.set_ylabel(r"$\langle A \rangle$")
    ax5.set_ylim(0, 1.05*AM_sol.y[AM.N_p:AM.N].max())
    for i in range(AM.N_p, AM.N):
        ax5.plot(AM_sol.t, AM_sol.y[i])

    ax6.set_xlabel("Time")
    ax6.set_ylabel(r"$\langle A \rangle$")
    ax6.set_ylim(0, 1.05*VM_sol.y[VM.N_p:VM.N].max())
    for i in range(VM.N_p, VM.N):
        ax6.plot(VM_sol.t, VM_sol.y[i])

    plt.savefig(f"figures/time_sol_AM_BM_VM_dA.png", format="png", dpi=500)


def plot_time_sol_mean(model, dA=None, save_fig=False):

    P_mean = model.y[:model.N_p].mean(axis=0)
    A_mean = model.y[model.N_p:model.N].mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
    title = f"Time solution {repr(model)}"
    if dA is not None:
        title += f" $d_{{A}} = {dA}$"
    fig.suptitle(title)

    ax1.plot(model.t, P_mean)
    ax1.set_ylabel("Plants")
    ax1.set_xlabel("Time")

    ax2.plot(model.t, A_mean)
    ax2.set_ylabel("Plants")
    ax2.set_xlabel("Time")

    if save_fig:
        plt.savefig(f"figures/time_sol_mean_{repr(model)}.png", format="png", dpi=500)

    return


def plot_time_sol_number_alive(model, dA=None, save_fig=False):

    P_alive = model.y_alive[:model.N_p].sum(axis=0)
    A_alive = model.y_alive[model.N_p:model.N].sum(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
    title = f"Time solution {repr(model)}"
    if dA is not None:
        title += f" $d_{{A}} = {dA}$"
    fig.suptitle(title)

    ax1.plot(model.t, P_alive)
    ax1.set_ylabel("# Plants species alive")
    ax1.set_xlabel("Time")
    ax1.set_ylim(0, model.N_p+1)

    ax2.plot(model.t, A_alive)
    ax2.set_ylabel("# Pollinator species alive")
    ax2.set_xlabel("Time")
    ax2.set_ylim(0, model.N_a+1)

    if save_fig:
        plt.savefig(f"figures/time_sol_number_alive_{repr(model)}.png", format="png", dpi=500)

    return


def plot_state_space_abundance_rate(model, fname, save_fig=False):

    with np.load(fname) as sol:
        rate_init = sol["rate_init"]
        abundance_init = sol["abundance_init"]
        dA_critical = sol["dA_critical"]

    nu = model.nu
    q = model.q
    G = model.G

    cmap = "plasma"

    extent = [rate_init.min(), rate_init.max(), abundance_init.min(), abundance_init.max()]
    aspect =  rate_init.max() / abundance_init.max()

    fig, axs = plt.subplots(constrained_layout=True)
    fig.suptitle(f"Critical driver of decline $d_A^*$: $nu = {nu}, q = {q}, G = {G}$")

    im = axs.matshow(
        dA_critical.T, cmap=cmap, origin="lower", extent=extent, aspect=aspect
    )
    axs.set_xlabel(r"$\lambda$ (rate of change of $d_A$)")
    axs.xaxis.set_ticks_position('bottom')
    axs.set_xticks(
        np.linspace(rate_init.min(), rate_init.max(), 6),
        np.linspace(rate_init.min(), rate_init.max(), 6)
    )
    axs.set_yticks(
        np.linspace(abundance_init.min(), abundance_init.max(), 6),
        np.linspace(abundance_init.min(), abundance_init.max(), 6)
    )
    axs.xaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    axs.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    axs.set_ylabel(r"Initial abundance per species")

    plt.colorbar(im, ax=axs)

    if save_fig:
        plt.savefig(
            f"figures/state_space_abundance_rate_{repr(model)}_nu{nu}_q{q}_G{G}.png",
            format="png", dpi=500
        )


def plot_state_space_abundance_env(
    model, fname, save_fig=False, extinct_threshold=None
):

    with np.load(fname) as sol:
        dA_init = sol["dA_init"]
        abundance_init = sol["abundance_init"]
        final_abundance = sol["final_abundance"]

    nu = model.nu
    q = model.q
    G = model.G

    if extinct_threshold is not None:
        final_abundance[final_abundance < extinct_threshold] == 0
        final_abundance[final_abundance >= extinct_threshold] == 1

    cmap = "plasma"

    extent = [dA_init.min(), dA_init.max(), abundance_init.min(), abundance_init.max()]
    aspect =  dA_init.max() / abundance_init.max()

    fig, axs = plt.subplots(figsize=(6, 6), constrained_layout=True)
    fig.suptitle(f"Final abundance: $nu = {nu}, q = {q}, G = {G}$")

    im = axs.matshow(
        final_abundance.T, cmap=cmap, origin="lower", extent=extent, aspect=aspect
    )
    axs.set_xlabel(r"$d_A$")
    axs.xaxis.set_ticks_position('bottom')
    axs.set_xticks(
        np.linspace(dA_init.min(), dA_init.max(), 6),
        np.linspace(dA_init.min(), dA_init.max(), 6)
    )
    axs.set_yticks(
        np.linspace(abundance_init.min(), abundance_init.max(), 6),
        np.linspace(abundance_init.min(), abundance_init.max(), 6)
    )
    axs.xaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    axs.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    axs.set_ylabel(r"Initial abundance per species")

    plt.colorbar(im, ax=axs)

    if save_fig:
        plt.savefig(
            f"figures/state_space_abundance_env_{repr(model)}_nu{nu}_q{q}_G{G}.png",
            format="png", dpi=500
        )


def plot_state_space_abundance_env_rate(
    model, fname, save_fig=False, extinct_threshold=None, relative_dA=False
):

    with np.load(fname) as sol:
        dA_init = sol["dA_init"]
        abundance_init = sol["abundance_init"]
        rate_init = sol["rate_init"]
        dA_critical = sol["dA_critical"]

    if len(abundance_init) == 4:

        if relative_dA:
            dA_critical[:, 0, :] = dA_critical[:, 0, :] - dA_init[:, np.newaxis]
            dA_critical[:, 1, :] = dA_critical[:, 1, :] - dA_init[:, np.newaxis]
            dA_critical[:, 2, :] = dA_critical[:, 2, :] - dA_init[:, np.newaxis]
            dA_critical[:, 3, :] = dA_critical[:, 3, :] - dA_init[:, np.newaxis]

        nu = model.nu
        q = model.q
        G = model.G

        aspect = dA_init[-1] / rate_init[-1]
        cmap = "plasma"
        vmin = dA_critical.min()
        vmax = dA_critical.max()

        extent = [rate_init.min(), rate_init.max(), dA_init.min(), dA_init.max()]
        aspect =  rate_init.max() / dA_init.max()

        fig, axs = plt.subplots(
            2, 2, sharex=True, sharey=True, constrained_layout=True
        )
        ((ax1, ax2), (ax3, ax4)) = axs
        if relative_dA:
            fig.suptitle(
                "Relative critical driver of decline $d_A^*$ ($d_A^*$ - $d_{A_{\text{init}}}$)\n"
                f"$nu = {nu}, q = {q}, G = {G}$"
            )
        else:
            fig.suptitle(
                f"Absolute critical driver of decline $d_A^*$\n$nu = {nu}, q = {q}, G = {G}$"
            )

        ax1.set_title(f"Init abundance $={abundance_init[0]}$")
        im = ax1.matshow(
            dA_critical[:, 0, :], cmap=cmap, origin="lower", extent=extent, aspect=aspect,
            vmin=vmin, vmax=vmax
        )
        # ax1.xaxis.set_ticks_position('bottom')
        # ax1.yaxis.set_ticks_position('left')
        ax1.set_ylabel(r"Initial $d_A$")
        ax1.set_yticks(
            np.linspace(dA_init.min(), dA_init.max(), 5)
        )
        ax1.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))

        ax2.set_title(f"Init abundance $={abundance_init[1]}$")
        im = ax2.matshow(
            dA_critical[:, 1, :], cmap=cmap, origin="lower", extent=extent, aspect=aspect,
            vmin=vmin, vmax=vmax
        )
        # ax2.xaxis.set_ticks_position('bottom')
        # ax2.yaxis.set_ticks_position('left')

        ax3.set_title(f"Init abundance $={abundance_init[2]}$")
        im = ax3.matshow(
            dA_critical[:, 2, :], cmap=cmap, origin="lower", extent=extent, aspect=aspect,
            vmin=vmin, vmax=vmax
        )
        ax3.xaxis.set_ticks_position('bottom')
        # ax3.yaxis.set_ticks_position('left')
        ax3.set_xlabel(r"$\lambda$")
        ax3.set_ylabel(r"Initial $d_A$")
        ax3.set_xticks(
            np.linspace(rate_init.min(), rate_init.max(), 5)
        )
        ax3.set_yticks(
            np.linspace(dA_init.min(), dA_init.max(), 5)
        )
        ax3.xaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
        ax3.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))

        ax4.set_title(f"Init abundance $={abundance_init[3]}$")
        im = ax4.matshow(
            dA_critical[:, 3, :], cmap=cmap, origin="lower", extent=extent, aspect=aspect,
            vmin=vmin, vmax=vmax
        )
        ax4.xaxis.set_ticks_position('bottom')
        # ax4.yaxis.set_ticks_position('left')
        ax4.set_xlabel(r"$\lambda$")
        ax4.set_xticks(
            np.linspace(rate_init.min(), rate_init.max(), 6)
        )
        ax4.xaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))

        # plt.colorbar(im, ax=axs)
        fig.colorbar(im, ax=axs)

        if save_fig:
            if relative_dA:
                plt.savefig(
                    f"figures/state_space_abundance_env_rate_{repr(model)}_nu{nu}_q{q}_G{G}_relative.png",
                    format="png", dpi=500
                )
            else:
                plt.savefig(
                    f"figures/state_space_abundance_env_rate_{repr(model)}_nu{nu}_q{q}_G{G}.png",
                    format="png", dpi=500
                )


def _add_arrow(line, position=None, direction="right", size=15, color=None):
    """Add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.

    Source:
    https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()

    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == "right":
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="-|>", color=color),
        size=size
    )
