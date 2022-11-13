#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 19/03/2022
# ---------------------------------------------------------------------------
""" experiments.py

Experiments
"""
# ---------------------------------------------------------------------------
import copy
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from tqdm import tqdm

import visualization as vis
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

MODELS = {
    "BM": pc.BaseModel,
    "AM": pc.AdaptiveModel
}


def state_space_rate_dA(fname, AM, rates=None, dAs_init=None, A_init=None):

    if rates is None:
        rates = np.linspace(0.0001, 0.1, 11)
    if dAs_init is None:
        dAs_init = np.linspace(0, 4, 11)
    if A_init is None:
        A_init = 1

    def dA_rate(t, r, dA_init):
        return dA_init + r * t

    t_end = int(1e5)
    n_steps = int(1e6) # number of interpolated time steps
    extinct_threshold = 0.01

    dAs_critical = np.zeros((len(rates), len(dAs_init)))

    curr_iter = 0
    total_iter = len(rates) * len(dAs_init)

    for i, rate in enumerate(rates):
        for j, dA_init in enumerate(dAs_init):
            print(f"Iteration {curr_iter + 1} out of {total_iter}")

            # initial conditions
            y0 = np.full(AM.N, A_init, dtype=float)
            y0 = np.concatenate((y0, copy.deepcopy(AM.alpha.flatten())))

            # drivers of decline
            dA = {
                "func": dA_rate,
                "args": (rate, dA_init)
            }

            sol = AM.solve(
                t_end, dA=dA, n_steps=n_steps, y0=y0, stop_on_collapse=True,
                extinct_threshold=extinct_threshold
            )

            # check if point of collapse has been found:
            if sol.status == 1:

                # find point of collapse
                A = AM.y[AM.N_p:AM.N]

                # put default at -1 if no population went extinct
                try:
                    ind = (A < extinct_threshold).all(axis=0).nonzero()[0][0]
                    t_extinct = AM.t[ind]
                    dAs_critical[i, j] = dA["func"](t_extinct, rate, dA_init)
                except IndexError:
                    dAs_critical[i, j] = -1
            else:
                dAs_critical[i, j] = -1

            curr_iter += 1

        np.savez(
            fname, rates=rates, dAs_init=dAs_init, A_init=A_init,
            dAs_critical=dAs_critical,
        )


def state_space_abundance_dA(fname, AM, dAs_init=None, A_init=None):

    if dAs_init is None:
        dAs_init = np.linspace(0, 4, 41)
    if A_init is None:
        A_init = np.linspace(0, 1, 11)

    t_end = int(1e5)
    n_steps = int(1e4) # number of interpolated time steps

    final_abundance = np.zeros((len(dAs_init), len(A_init)))

    curr_iter = 0
    total_iter = len(dAs_init) * len(A_init)

    for i, dA in enumerate(dAs_init):
        for j, abundance in enumerate(A_init):
            print(f"Iteration {curr_iter + 1} out of {total_iter}")

            # initial conditions
            y0 = np.full(AM.N, abundance, dtype=float)
            y0 = np.concatenate((y0, copy.deepcopy(AM.alpha.flatten())))

            sol = AM.solve(
                t_end, dA=dA, n_steps=n_steps, y0=y0, stop_on_collapse=False,
                stop_on_equilibrium=True
            )

            A_mean = AM.y[AM.N_p:AM.N].mean(axis=0)

            # final abundace is the mean abundace at the final time point
            final_abundance[i, j] = A_mean[-1]

            curr_iter += 1

        np.savez(
            fname, dAs_init=dAs_init, A_init=A_init, final_abundance=final_abundance,
        )


def state_space_abundance_rate_critical_dA(fname, AM, rates=None, A_init=None):

    if rates is None:
        rates = np.linspace(0.0001, 0.1, 11)
    if A_init is None:
        A_init = np.linspace(0, 1, 11)

    def dA_rate(t, r):
        return r * t

    t_end = int(1e5)
    n_steps = int(1e4) # number of interpolated time steps
    extinct_threshold = 0.01

    dAs_critical = np.zeros((len(rates), len(A_init)))

    curr_iter = 0
    total_iter = len(rates) * len(A_init)

    for i, rate in enumerate(rates):
        for j, abundance in enumerate(A_init):
            print(f"Iteration {curr_iter + 1} out of {total_iter}")

            # initial conditions
            y0 = np.full(AM.N, abundance, dtype=float)
            y0 = np.concatenate((y0, copy.deepcopy(AM.alpha.flatten())))

            # drivers of decline
            dA = {
                "func": dA_rate,
                "args": (rate, )
            }

            sol = AM.solve(
                t_end, dA=dA, n_steps=n_steps, y0=y0, stop_on_collapse=True,
                extinct_threshold=extinct_threshold
            )

            # check if point of collapse has been found:
            if sol.status == 1:

                # find point of collapse
                A_mean = AM.y[AM.N_p:AM.N].mean(axis=0)

                # put default at -1 if no population went extinct
                try:
                    ind = (A_mean < extinct_threshold).nonzero()[0][0]
                    t_extinct = AM.t[ind]
                    dAs_critical[i, j] = dA["func"](t_extinct, rate)
                except IndexError:
                    dAs_critical[i, j] = -1
            else:
                dAs_critical[i, j] = -1

            curr_iter += 1

        np.savez(
            fname, rates=rates, A_init=A_init, dAs_critical=dAs_critical,
        )


def state_space_abundance_rate_critical_dA_all(fname, AM, rates=None, A_init=None):

    if rates is None:
        rates = np.linspace(0.0001, 0.1, 11)
    if A_init is None:
        A_init = np.linspace(0, 1, 11)

    def dA_rate(t, r):
        return r * t

    t_end = int(1e5)
    n_steps = int(1e6) # number of interpolated time steps
    extinct_threshold = 0.01

    dAs_critical = np.zeros((len(rates), len(A_init)))

    curr_iter = 0
    total_iter = len(rates) * len(A_init)

    for i, rate in enumerate(rates):
        for j, abundance in enumerate(A_init):
            print(f"Iteration {curr_iter + 1} out of {total_iter}")

            # initial conditions
            y0 = np.full(AM.N, abundance, dtype=float)
            y0 = np.concatenate((y0, copy.deepcopy(AM.alpha.flatten())))

            # drivers of decline
            dA = {
                "func": dA_rate,
                "args": (rate, )
            }

            sol = AM.solve(
                t_end, dA=dA, n_steps=n_steps, y0=y0, stop_on_collapse=True,
                extinct_threshold=extinct_threshold
            )

            # check if point of collapse has been found:
            if sol.status == 1:

                # find point of collapse
                A = AM.y[AM.N_p:AM.N]

                # print((A[:, -1] < extinct_threshold).all())
                # print((A < extinct_threshold).all(axis=0))
                # if not (A[:, -1] < extinct_threshold).all():
                #     print(A[:, -1])
                #     print(AM.t[-1])
                # put default at -1 if no population went extinct
                try:
                    ind = (A < extinct_threshold).all(axis=0).nonzero()[0][0]
                    t_extinct = AM.t[ind]
                    dAs_critical[i, j] = dA["func"](t_extinct, rate)
                except IndexError:
                    dAs_critical[i, j] = -1
            else:
                dAs_critical[i, j] = -1

            curr_iter += 1

        np.savez(
            fname, rates=rates, A_init=A_init, dAs_critical=dAs_critical,
        )


def state_space_rate_critical_dA(fname, AM, rates=None, A_init=None):

    if rates is None:
        rates = np.linspace(0.0001, 0.1, 11)
    if A_init is None:
        A_init = [0.2]

    def dA_rate(t, r):
        return r * t

    t_end = int(1e5)
    n_steps = int(1e6) # number of interpolated time steps
    extinct_threshold = 0.01

    dAs_critical = np.zeros((len(rates), len(A_init)))
    curr_iter = 0
    total_iter = len(rates) * len(A_init)

    for i, rate in enumerate(rates):
        for j, abundance in enumerate(A_init):
            print(f"Iteration {curr_iter + 1} out of {total_iter}")

            # initial conditions
            y0 = np.full(AM.N, abundance, dtype=float)
            y0 = np.concatenate((y0, copy.deepcopy(AM.alpha.flatten())))

            # drivers of decline
            dA = {
                "func": dA_rate,
                "args": (rate, )
            }

            sol = AM.solve(
                t_end, dA=dA, n_steps=n_steps, y0=y0, stop_on_collapse=True,
                extinct_threshold=extinct_threshold
            )

            # check if point of collapse has been found:
            if sol.status == 1:

                # find point of collapse
                A = AM.y[AM.N_p:AM.N]

                # print((A[:, -1] < extinct_threshold).all())
                # print((A < extinct_threshold).all(axis=0))
                # if not (A[:, -1] < extinct_threshold).all():
                #     print(A[:, -1])
                #     print(AM.t[-1])
                # put default at -1 if no population went extinct
                try:
                    ind = (A < extinct_threshold).all(axis=0).nonzero()[0][0]
                    t_extinct = AM.t[ind]
                    dAs_critical[i, j] = dA["func"](t_extinct, rate)
                except IndexError:
                    dAs_critical[i, j] = -1
            else:
                dAs_critical[i, j] = -1

            curr_iter += 1
    np.savez(
        fname, rates=rates, A_init=A_init, dAs_critical=dAs_critical,
    )


def hysteresis_q(AM, dAs=None, qs=None, seed=None, fnumber=0):
    """Calculate hysteresis as function of dA for different q. """
    if dAs is None:
        dAs = np.linspace(0, 4, 21)
    if qs is None:
        qs = np.linspace(0, 1, 11)
    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]
    rng = np.random.default_rng(seed)

    for i, q in enumerate(qs):
        print(f"\nCalculating q: {i+1} out of {len(qs)}...")

        # set q parameter, set seed and rng
        AM.q = q
        AM.rng = rng

        fname = f"output/hysteresis_G{AM.G}_nu{AM.nu}_q{AM.q}_{fnumber}"

        hysteresis(fname, AM, dAs=dAs)


def find_dA_collapse_recovery(fname, AM, dA_step=0.02):
    """Calculate hysteresis as function of dA. """
    # maximum simulation time
    t_end = int(1e4)
    t_step = int(1e4)
    extinct_threshold = 0.01

    # obtain initial solution
    y0 = AM.equilibrium()

    if AM.is_all_alive()[-1]:
        is_feasible = True
    else:
        is_feasible = False

    # calculate solution for increasing dA
    print("\nCalculating hysteresis forward...")
    dA = 0
    while (AM.y[AM.N_p:AM.N, -1] > extinct_threshold).any():

        dA += dA_step
        AM.solve(t_step, dA=dA, y0=y0, save_period=0, stop_on_equilibrium=True)


        y0 = AM.y[:, -1]
        y0 = np.concatenate((y0, AM.y_partial[:, -1]))

    dA_collapse = dA

    # calculate solution for decreasing dA
    print("\nCalculating hysteresis backward...")
    while not (AM.y[AM.N_p:AM.N, -1] > extinct_threshold).any():

        dA -= dA_step
        AM.solve(t_step, dA=dA, y0=y0, save_period=0, stop_on_equilibrium=True)

        y0 = AM.y[:, -1]
        y0 = np.concatenate((y0, AM.y_partial[:, -1]))

    dA_recover = dA

    np.savez(
        fname, dA_collapse=dA_collapse, dA_recover=dA_recover, is_feasible=is_feasible
    )


def hysteresis(fname, AM, dAs=None):
    """Calculate hysteresis as function of dA. """
    # maximum simulation time
    t_end = 1000
    t_step = 100

    if dAs is None:
        dAs = np.linspace(0, 4, 21)

    # save only steady state solutions
    P_sol_forward = np.zeros((len(dAs), AM.N_p))
    A_sol_forward = np.zeros((len(dAs), AM.N_a))
    P_sol_backward = np.zeros((len(dAs), AM.N_p))
    A_sol_backward = np.zeros((len(dAs), AM.N_a))

    # obtain initial solution
    AM.solve(t_end, dA=0, save_period=0, stop_on_equilibrium=True)

    if AM.is_all_alive()[-1]:
        is_feasible = True
    else:
        is_feasible = False

    # calculate solution for increasing dA
    print("\nCalculating hysteresis forward...")
    for i, dA in enumerate(dAs):
        print(f"{i+1} out of {len(dAs)}...")

        y0 = AM.y[:, -1]
        y0 = np.concatenate((y0, AM.y_partial[:, -1]))

        AM.solve(t_step, dA=dA, y0=y0, save_period=0, stop_on_equilibrium=True)

        P_sol_forward[i] = AM.y[:AM.N_p, -1]
        A_sol_forward[i] = AM.y[AM.N_p:AM.N, -1]

    # calculate solution for decreasing dA
    print("\nCalculating hysteresis backward...")
    for i, dA in enumerate(np.flip(dAs)):
        print(f"{i+1} out of {len(dAs)}...")

        y0 = AM.y[:, -1]
        y0 = np.concatenate((y0, AM.y_partial[:, -1]))

        AM.solve(t_step, dA=dA, y0=y0, save_period=0, stop_on_equilibrium=True)

        P_sol_backward[i] = AM.y[:AM.N_p, -1]
        A_sol_backward[i] = AM.y[AM.N_p:AM.N, -1]

    np.savez(
        fname, dAs=dAs, P_sol_forward=P_sol_forward, A_sol_forward=A_sol_forward,
        P_sol_backward=P_sol_backward, A_sol_backward=A_sol_backward,
        is_feasible=is_feasible
    )


def hysteresis_rate(fname, AM, rate=0.001, dA_max=3):

    # maximum simulation time
    t_equilibrium = 1000

    # time needed to reach dA_max for given rate
    t_end = dA_max / rate
    n_steps = 1000

    # save only steady state solutions
    P_sol_forward = np.zeros((n_steps, AM.N_p))
    A_sol_forward = np.zeros((n_steps, AM.N_a))
    P_sol_backward = np.zeros((n_steps, AM.N_p))
    A_sol_backward = np.zeros((n_steps, AM.N_a))

    if AM.is_all_alive()[-1]:
        is_feasible = True
    else:
        is_feasible = False

    # obtain initial solution
    AM.solve(
        t_equilibrium, n_steps=1000, dA=0, save_period=0, stop_on_equilibrium=True
    )

    # calculate solution for increasing dA
    print("\nCalculating hysteresis forward...")
    dA_rate = {
        "func": lambda t, rate: rate * t,
        "args": (rate, )
    }
    y0 = AM.y[:, -1]
    y0 = np.concatenate((y0, AM.y_partial[:, -1]))
    AM.solve(
        t_end, y0=y0, n_steps=n_steps, dA=dA_rate, save_period=0
    )
    P_sol_forward = AM.y[:AM.N_p].T
    A_sol_forward = AM.y[AM.N_p:AM.N].T

    dAs_forward = rate * AM.t

    # calculate solution for decreasing dA
    print("\nCalculating hysteresis backward...\n")
    def func(t, rate, dA_max):
        if dA_max - rate * t < 0:
            return 0
        else:
            return dA_max - rate * t
    dA_rate = {
        "func": func,
        "args": (rate, dA_max)
    }
    y0 = AM.y[:, -1]
    y0 = np.concatenate((y0, AM.y_partial[:, -1]))

    # make sure system ends up in final equilibrium
    AM.solve(
        t_end*10, y0=y0, n_steps=n_steps, dA=dA_rate, save_period=0,
        stop_on_equilibrium=True
    )
    P_sol_backward = AM.y[:AM.N_p].T
    A_sol_backward = AM.y[AM.N_p:AM.N].T

    dAs_backward = [func(t, rate, dA_max) for t in AM.t]

    np.savez(
        fname, dAs_forward=dAs_forward, dAs_backward=dAs_backward,
        P_sol_forward=P_sol_forward, A_sol_forward=A_sol_forward,
        P_sol_backward=P_sol_backward, A_sol_backward=A_sol_backward,
        is_feasible=is_feasible, rate=rate
    )


#################################################################################
### Old experiments
#################################################################################

def AM_baseline_comp(dA=0, plot_network=False, save_fig=False):
    """Compares AdaptiveModel (AM) with AdaptiveModel baseline (AM_base).
    The foraging effort is fixed for the AM_base
    """
    # create two identical rng's for fair comparison between BM and AM
    seed_seq = np.random.SeedSequence()
    AM_base_rng = np.random.default_rng(seed_seq)
    AM_rng = np.random.default_rng(seed_seq)

    N_p = 20
    N_a = 20
    mu = 0
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_end = 50

    AM_base = pc.AdaptiveModel (
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=AM_base_rng,
        G=0
    )
    # # for baseline make alpha * beta = gamma
    # AM_base.gamma_trade_off = 0.5
    # AM_base.gamma = AM_base._generate_gamma()
    # AM_base.alpha = AM_base.gamma / AM_base.gamma.sum(axis=0)
    # AM_base.beta = AM_base.network
    AM = pc.AdaptiveModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=AM_rng, G=1
    )

    AM_base_sol = AM_base.solve(t_end, dA=dA)
    AM_sol = AM.solve(t_end, dA=dA, save_period=0)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, constrained_layout=True, sharex=True, sharey=True
    )
    fig.suptitle(f"Baseline comparison AM $d_A = {dA}$")

    ax1.set_title("Fixed Foraging")
    ax1.set_ylabel("Plant abundance")
    for i in range(AM_base.N_p):
        line_polls, = ax1.plot(AM_base_sol.t, AM_base_sol.y[i])

    ax2.set_title("Adaptive Foraging")
    for i in range(AM.N_p):
        line_polls, = ax2.plot(AM_sol.t, AM_sol.y[i])

    ax3.set_xlabel("Time")
    ax3.set_ylabel("Pollinator abundance")
    for j in range(AM_base.N_p, AM_base.N):
        line_polls, = ax3.plot(AM_base_sol.t, AM_base_sol.y[j])

    ax4.set_xlabel("Time")
    for j in range(AM.N_p, AM.N):
        line_polls, = ax4.plot(AM_sol.t, AM_sol.y[j])

    if save_fig:
        plt.savefig(f"figures/AM_baseline_comp_dA_{dA}.png", format="png", dpi=500)

    if plot_network:
        alpha = AM_base_sol.y_partial[:, -1].reshape((AM_base.N_p, AM_base.N_a))
        beta = AM_base.beta
        network = AM_base.network
        forbidden_network = AM_base.forbidden_network
        title = f"Final network FM r = 0"
        vis.plot_alpha_beta_network(network, alpha, beta, forbidden_network, title=title)

        alpha = AM_sol.y_partial[:, -1].reshape((AM.N_p, AM.N_a))
        beta = AM.beta
        network = AM.network
        forbidden_network = AM.forbidden_network
        title = f"Final network AM r = 0"
        vis.plot_alpha_beta_network(network, alpha, beta, forbidden_network, title=title)

    return


def AM_four_dAs(dAs=[0, 1, 2, 3], G=1, save_fig=False, rng=None, plot_network=False, t_end=100):

    if rng is None:
        rng = np.random.default_rng(seed)

    N_p = 20
    N_a = 20
    mu = 0
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"

    AM = pc.AdaptiveModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=rng, G=G
    )
    AM_sols = []
    for dA in dAs:
        AM_sols.append(AM.solve(t_end, dA=dA, save_period=0, n_steps=int(1e4)))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, constrained_layout=True, sharex=True, sharey=True
    )
    if G > 0:
        fig.suptitle(f"Adaptive Foraging Pollinators")
    else:
        fig.suptitle(f"Fixed Foraging Pollinators")

    def set_title(i):
        if isinstance(dAs[i], (int, float)):
            title = f"dA = {dAs[i]}"
        elif isinstance(dAs[i], (dict)):
            r = dAs[i].get("args", None)
            title = f"r = {r}"
        return title

    ax1.set_title(set_title(0))
    ax1.set_ylabel("Pollinator abundance")
    AM_sol = AM_sols[0]
    for j in range(AM.N_p, AM.N):
        line_polls, = ax1.plot(AM_sol.t, AM_sol.y[j])

    ax2.set_title(set_title(1))
    AM_sol = AM_sols[1]
    for j in range(AM.N_p, AM.N):
        line_polls, = ax2.plot(AM_sol.t, AM_sol.y[j])

    ax3.set_title(set_title(2))
    ax3.set_ylabel("Pollinator abundance")
    AM_sol = AM_sols[2]
    for j in range(AM.N_p, AM.N):
        line_polls, = ax3.plot(AM_sol.t, AM_sol.y[j])

    ax4.set_title(set_title(3))
    AM_sol = AM_sols[3]
    for j in range(AM.N_p, AM.N):
        line_polls, = ax4.plot(AM_sol.t, AM_sol.y[j])

    if save_fig:
        plt.savefig(f"figures/AM_polls_four_dAs_{dA}.png", format="png", dpi=500)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, constrained_layout=True, sharex=True, sharey=True
    )
    if G > 0:
        fig.suptitle(f"Adaptive Foraging Plants")
    else:
        fig.suptitle(f"Fixed Foraging Plants")

    ax1.set_title(set_title(0))
    ax1.set_ylabel("Plant abundance")
    AM_sol = AM_sols[0]
    for i in range(AM.N_p):
        line_polls, = ax1.plot(AM_sol.t, AM_sol.y[i])

    ax2.set_title(set_title(1))
    AM_sol = AM_sols[1]
    for i in range(AM.N_p):
        line_polls, = ax2.plot(AM_sol.t, AM_sol.y[i])

    ax3.set_title(set_title(2))
    ax3.set_ylabel("Plant abundance")
    AM_sol = AM_sols[2]
    for i in range(AM.N_p):
        line_polls, = ax3.plot(AM_sol.t, AM_sol.y[i])

    ax4.set_title(set_title(3))
    AM_sol = AM_sols[3]
    for i in range(AM.N_p):
        line_polls, = ax4.plot(AM_sol.t, AM_sol.y[i])

    if save_fig:
        plt.savefig(f"figures/AM_plants_four_dAs_{dA}.png", format="png", dpi=500)

    if plot_network:
        alpha = AM_sols[0].y_partial[:, -1].reshape((AM.N_p, AM.N_a))
        beta = AM.beta
        network = AM.network
        forbidden_network = AM.forbidden_network
        if G > 0:
            title = f"Final network AM r = 0"
        else:
            title = f"Final network FM r = 0"
        vis.plot_alpha_beta_network(network, alpha, beta, forbidden_network, title=title)

    return


def BM_AM_baseline_comp(dA=0, save_fig=False):
    """Compares BaseModel (BM) with AdaptiveModel (AM).
    The foraging effort is fixed for the AM, it also ensure that alpha * beta = gamma
    """
    # create two identical rng's for fair comparison between BM and AM
    seed_seq = np.random.SeedSequence()
    BM_rng = np.random.default_rng(seed_seq)
    AM_rng = np.random.default_rng(seed_seq)

    N_p = 20
    N_a = 20
    mu = 0
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_end = 50

    BM = pc.BaseModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=BM_rng
    )
    AM = pc.AdaptiveModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=AM_rng, G=1
    )
    # for baseline comparison make alpha * beta = gamma
    AM.gamma_trade_off = 0.5
    AM.gamma = AM._generate_gamma()
    AM.alpha = AM.gamma / AM.gamma.sum(axis=0)
    AM.beta = AM.network

    BM_sol = BM.solve(t_end, dA=dA)
    AM_sol = AM.solve(t_end, dA=dA, save_period=0)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, constrained_layout=True, sharex=True, sharey=True
    )
    fig.suptitle(f"Baseline comparison BM and AM $d_A = {dA}$")

    ax1.set_title("BaseModel")
    ax1.set_ylabel("Plant abundance")
    for i in range(BM.N_p):
        line_polls, = ax1.plot(BM_sol.t, BM_sol.y[i])

    ax2.set_title("AdaptiveModel")
    for i in range(AM.N_p):
        line_polls, = ax2.plot(AM_sol.t, AM_sol.y[i])

    ax3.set_xlabel("Time")
    ax3.set_ylabel("Pollinator abundance")
    for j in range(BM.N_p, BM.N):
        line_polls, = ax3.plot(BM_sol.t, BM_sol.y[j])

    ax4.set_xlabel("Time")
    for j in range(AM.N_p, AM.N):
        line_polls, = ax4.plot(AM_sol.t, AM_sol.y[j])

    if save_fig:
        plt.savefig(f"figures/BM_AM_baseline_comp_dA_{dA}.png", format="png", dpi=500)

    return


def AM_beta_alpha(dA=0, seed=None, save_fig=False):

    if seed is None:
        seed = np.random.SeedSequence()

    rng = np.random.default_rng(seed)

    N_p = 20
    N_a = 20
    mu = 0.0001
    nu = 0.3
    connectance = 0.2
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_end = 500

    AM = pc.AdaptiveModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, rng=rng, seed=seed, nu=nu
    )
    AM_sol = AM.solve(t_end, dA=dA)

    alpha_init = copy.deepcopy(AM.alpha)
    alpha_end = copy.deepcopy(AM_sol.y_partial[:, -1].reshape((AM.N_p, AM.N_a)))
    beta = copy.deepcopy(AM.beta)

    title = r"Adaptive Model: nested $\beta$ normalized"
    return vis.plot_AM_alpha_init_alpha_end_beta(
        alpha_init, alpha_end, beta, save_fig=save_fig, title=title
    )


def VM_alpha(dA=0, seed=None, save_fig=False):

    if seed is None:
        seed = np.random.SeedSequence()

    rng = np.random.default_rng(seed)

    N_p = 20
    N_a = 20
    mu = 0.0001
    nu = 0.3
    connectance = 0.2
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_end = 5000

    VM = pc.ValdovinosModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, rng=rng, seed=seed
    )
    VM_sol = VM.solve(t_end, dA=dA)

    alpha_init = copy.deepcopy(VM.alpha0)
    alpha_end = copy.deepcopy(VM_sol.y_partial[:, -1].reshape((VM.N_p, VM.N_a)))

    title = r"Valdovinos Model"
    return vis.plot_VM_alpha_init_alpha_end(
        alpha_init, alpha_end, save_fig=save_fig, title=title
    )


def time_sol_AM_BM_VM(dA=0, seed=None, save_fig=False):

    if seed is None:
        seed = np.random.SeedSequence()

    rng = np.random.default_rng(seed)

    N_p = 20
    N_a = 20
    mu = 0.0001
    nu = 0.01
    connectance = 0.2
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    dA = 0

    AM = pc.AdaptiveModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, rng=rng, seed=seed, nu=nu
    )
    AM_sol = AM.solve(100, dA=dA, n_steps=int(1e5))

    BM = pc.BaseModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, rng=rng, seed=seed
    )
    BM_sol = BM.solve(100, dA=dA, n_steps=int(1e5))

    VM = pc.ValdovinosModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, rng=rng, seed=seed
    )
    VM_sol = VM.solve(10000, dA=dA, n_steps=int(1e5))

    vis.plot_time_sol_AM_BM_VM(
        AM_sol, AM, BM_sol, BM, VM_sol, VM, dA=dA, save_fig=save_fig
    )

    return


def AM_q(seed=None, save_fig=False, recalculate=True):

    G = 0.1
    nu = 0.5
    fname = f"output/AM_q_G{G}_nu{nu}"

    if recalculate:
        if seed is None:
            seed = np.random.SeedSequence()


        qs = np.linspace(0, 0.7, 16)

        dAs_all = []
        P_mean_all = []
        P_backward_mean_all = []
        A_mean_all = []
        A_backward_mean_all = []
        is_feasible_all = []

        for i, q in enumerate(qs):
            print(f"{i+1} out of {len(qs)}...\n")

            # run model everytime with same seed
            dAs, P_mean, A_mean, P_backward_mean, A_backward_mean, is_feasible = state_space_AM(
                seed=seed, G=G, nu=nu, q=q
            )

            dAs_all.append(dAs)
            P_mean_all.append(P_mean)
            P_backward_mean_all.append(P_backward_mean)
            A_mean_all.append(A_mean)
            A_backward_mean_all.append(A_backward_mean)
            is_feasible_all.append(is_feasible)

        dAs_all = np.asarray(dAs_all)
        P_mean_all = np.asarray(P_mean_all)
        P_backward_mean_all = np.asarray(P_backward_mean_all)
        A_mean_all = np.asarray(A_mean_all)
        A_backward_mean_all = np.asarray(A_backward_mean_all)

        np.savez(
            fname, qs=qs, dAs_all=dAs_all, P_mean_all=P_mean_all, A_mean_all=A_mean_all,
            P_backward_mean_all=P_backward_mean_all, A_backward_mean_all=A_backward_mean_all,
            is_feasible_all=is_feasible_all
        )

    with np.load(fname + ".npz") as sol:
        qs = sol["qs"]
        dAs_all = sol["dAs_all"]
        P_mean_all = sol["P_mean_all"]
        P_backward_mean_all = sol["P_backward_mean_all"]
        A_mean_all = sol["A_mean_all"]
        A_backward_mean_all = sol["A_backward_mean_all"]
        is_feasible_all = sol["is_feasible_all"]

    threshold = 0.01   # abundance at which we assume all species to be extinct
    tipping_forward = []
    q_forward = []
    tipping_backward = []
    q_backward = []

    for i, q in enumerate(qs):
        # find tipping points
        try:
            ind = (A_mean_all[i] <= threshold).nonzero()[0][0]
            tipping_forward.append(dAs_all[i][ind])
            q_forward.append(q)
        except IndexError:
            # no inds exist for which holds the condition A_mean <= threshold
            tipping_forward.append(0)
            q_forward.append(q)
        try:
            ind = (A_backward_mean_all[i] <= threshold).nonzero()[0][-1]
            tipping_backward.append(np.flip(dAs_all[i])[ind])
            q_backward.append(q)
        except IndexError:
            # no inds exist for which holds the condition A_mean <= threshold
            tipping_backward.append(0)
            q_backward.append(q)

    fig, axs = plt.subplots(constrained_layout=True)
    fig.suptitle(r"Value of $d_A$ at which all pollinator species go extinct")
    scatter_collapse = axs.scatter(
        q_forward, tipping_forward, marker="x", color="blue", label="Collapse"
    )
    scatter_recovery = axs.scatter(
        q_backward, tipping_backward, marker="^", color="red", label="Recovery"
    )
    y_min = min(min(tipping_backward), min(tipping_forward))
    y_max = max(max(tipping_backward), max(tipping_forward))
    width = abs(q_forward[1] - q_forward[0])/2
    for i, is_feasible in enumerate(is_feasible_all):
        if is_feasible:
            color = "#009E73"
        else:
            color = "#D55E00"
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
            facecolor="#D55E00", edgecolor="#D55E00", linewidth=0.1,
            label="No feasible network", alpha=0.2
        )
    ]
    axs.legend(handles=legend_elements)

    if save_fig:
        plt.savefig(f"figures/AM_dA_q_G{G}_nu{nu}.png", format="png", dpi=300)

    return


def state_space_AM_BM(seed=None, save_fig=False, recalculate=True):
    if recalculate:
        state_space_AM(
            G=0, nu=0.01, q=2, seed=seed, save_fig=save_fig, fname=f"output/AM.npz"
        )
        state_space_BM(
            seed=seed, save_fig=save_fig, fname=f"output/BM.npz"
        )
    return vis.plot_state_space_AM_BM(f"output/AM.npz", f"output/BM.npz")


def state_space_AM_BM_VM(seed=None, save_fig=False, recalculate=True):
    if recalculate:
        state_space_AM(
            G=1, seed=seed, save_fig=save_fig, fname=f"output/AM2.npz"
        )
        state_space_BM(
            seed=seed, save_fig=save_fig, fname=f"output/BM2.npz"
        )
        state_space_VM(
            seed=seed, save_fig=save_fig, fname=f"output/VM2.npz"
        )
    return vis.plot_state_space_AM_BM_VM(
        f"output/AM2.npz", f"output/BM2.npz", f"output/VM2.npz"
    )


def state_space_rate_AM_BM_VM(seed=None, save_fig=False, recalculate=True):
    if recalculate:
        state_space_rate_AM(
            G=1, seed=seed, save_fig=save_fig, fname=f"output/AM_rate_no_const.npz"
        )
        state_space_rate_BM(
            seed=seed, save_fig=save_fig, fname=f"output/BM_rate_no_const.npz"
        )
        state_space_rate_VM(
            seed=seed, save_fig=save_fig, fname=f"output/VM_rate_no_const.npz"
        )
    return vis.plot_state_space_rate_AM_BM_VM(
        f"output/AM_rate_no_const.npz", f"output/BM_rate_no_const.npz", f"output/VM_rate_no_const.npz"
    )
    # if recalculate:
    #     state_space_rate_AM(
    #         G=1, seed=seed, save_fig=save_fig, fname=f"output/AM_rate.npz"
    #     )
    #     state_space_rate_BM(
    #         seed=seed, save_fig=save_fig, fname=f"output/BM_rate.npz"
    #     )
    #     state_space_rate_VM(
    #         seed=seed, save_fig=save_fig, fname=f"output/VM_rate.npz"
    #     )
    # return vis.plot_state_space_rate_AM_BM_VM(
    #     f"output/AM_rate.npz", f"output/BM_rate.npz", f"output/VM_rate.npz"
    # )


def state_space_AM(G=1, nu=0.1, q=1, seed=None, plot=False, save_fig=False, fname=None):

    if seed is None:
        seed = np.random.SeedSequence()

    rng1 = np.random.default_rng(seed)

    N_p = 15
    N_a = 40
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_end = 200
    t_sol = 20

    dAs = np.linspace(0, 3, 31)

    # save only steady state solutions
    P_sol = np.zeros((len(dAs), N_p))
    A_sol = np.zeros((len(dAs), N_a))
    P_sol_backward = np.zeros((len(dAs), N_p))
    A_sol_backward = np.zeros((len(dAs), N_a))

    AM = pc.AdaptiveModel (
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, seed=seed,
        rng=rng1, G=G, nu=nu, q=q, feasible=True, feasible_iters=50
    )
    if AM.is_all_alive()[-1]:
        is_feasible = True
    else:
        is_feasible = False

    # obtain initial solution
    sol = AM.solve(t_end, dA=0, save_period=0)


    t0 = timer()
    for i, dA in enumerate(dAs):
        print(f"{i+1} out of {len(dAs)}...")

        y0 = sol.y[:, -1]
        y0 = np.concatenate((y0, sol.y_partial[:, -1]))

        sol = AM.solve(t_sol, dA=dA, y0=y0, save_period=0)

        P_sol[i] = sol.y[:AM.N_p, -1]
        A_sol[i] = sol.y[AM.N_p:AM.N, -1]

    print(f"Phase space forward. Time elapsed: {timer()-t0:.5f} seconds\n")

    t0 = timer()
    for i, dA in enumerate(np.flip(dAs)):
        print(f"{i+1} out of {len(dAs)}...")

        y0 = sol.y[:, -1]
        y0 = np.concatenate((y0, sol.y_partial[:, -1]))

        sol = AM.solve(t_sol, dA=dA, y0=y0, save_period=0)

        P_sol_backward[i] = sol.y[:AM.N_p, -1]
        A_sol_backward[i] = sol.y[AM.N_p:AM.N, -1]

    print(f"Phase space backward. Time elapsed: {timer()-t0:.5f} seconds\n")

    # calculate mean abundancies
    P_mean = np.mean(P_sol, axis=1)
    A_mean = np.mean(A_sol, axis=1)
    P_backward_mean = np.mean(P_sol_backward, axis=1)
    A_backward_mean = np.mean(A_sol_backward, axis=1)

    if plot:
        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("AM Phase space plants")
        axs.plot(dAs, P_mean, color="blue")
        axs.plot(np.flip(dAs), P_backward_mean, color="red")
        axs.set_xlabel(r"Driver of decline")
        axs.set_ylabel(r"$\langle P \rangle$")
        if save_fig:
            plt.savefig(f"figures/state_space_P_G_{G}.png", format="png", dpi=300)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("AM Phase space pollinators")
        axs.plot(dAs, A_mean, color="blue")
        axs.plot(np.flip(dAs), A_backward_mean, color="red")
        axs.set_xlabel(r"Driver of decline")
        axs.set_ylabel(r"$\langle A \rangle$")
        if save_fig:
            plt.savefig(f"figures/state_space_A_G_{G}.png", format="png", dpi=300)

    if fname is not None:
        np.savez(
            fname, dAs=dAs, P_mean=P_mean, A_mean=A_mean,
            P_backward_mean=P_backward_mean, A_backward_mean=A_backward_mean,
            is_feasible=is_feasible
        )

    return dAs, P_mean, A_mean, P_backward_mean, A_backward_mean, is_feasible


def state_space_ARM(G, seed=None, plot=False, save_fig=False, fname=None):

    if seed is None:
        seed = np.random.SeedSequence()

    rng1 = np.random.default_rng(seed)

    N_p = 20
    N_a = 20
    mu = 0.0001
    nu = 0.1
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_end = 500
    t_sol = 500

    dAs = np.linspace(0, 1, 50)

    # save only steady state solutions
    P_sol = np.zeros((len(dAs), N_p))
    A_sol = np.zeros((len(dAs), N_a))
    R_sol = np.zeros((len(dAs), N_p))
    P_sol_backward = np.zeros((len(dAs), N_p))
    A_sol_backward = np.zeros((len(dAs), N_a))
    R_sol_backward = np.zeros((len(dAs), N_p))

    ARM = pc.AdaptiveResourceModel (
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=rng1, seed=seed,
        G=G, nu=nu
    )

    # obtain initial solution
    sol = ARM.solve(t_end, dA=0, save_period=0)
    # vis.plot_time_sol_ARM(sol, ARM, dA=0)

    t0 = timer()
    for i, dA in enumerate(dAs):
        print(f"{i+1} out of {len(dAs)}...")

        y0 = sol.y[:, -1]

        y0 = np.concatenate((y0[:ARM.N], sol.y_partial[:, -1], y0[ARM.N:]))

        sol = ARM.solve(t_sol, dA=dA, y0=y0, save_period=0)

        P_sol[i] = sol.y[:ARM.N_p, -1]
        A_sol[i] = sol.y[ARM.N_p:ARM.N, -1]
        R_sol[i] = sol.y[ARM.N:, -1]

    print(f"Phase space forward. Time elapsed: {timer()-t0:.5f} seconds\n")

    t0 = timer()
    for i, dA in enumerate(np.flip(dAs)):
        print(f"{i+1} out of {len(dAs)}...")

        y0 = sol.y[:, -1]
        y0 = np.concatenate((y0[:ARM.N], sol.y_partial[:, -1], y0[ARM.N:]))

        sol = ARM.solve(t_sol   , dA=dA, y0=y0, save_period=0)

        P_sol_backward[i] = sol.y[:ARM.N_p, -1]
        A_sol_backward[i] = sol.y[ARM.N_p:ARM.N, -1]
        R_sol_backward[i] = sol.y[ARM.N:, -1]

    print(f"Phase space backward. Time elapsed: {timer()-t0:.5f} seconds\n")

    # calculate mean abundancies
    P_mean = np.mean(P_sol, axis=1)
    A_mean = np.mean(A_sol, axis=1)
    R_mean = np.mean(R_sol, axis=1)
    P_backward_mean = np.mean(P_sol_backward, axis=1)
    A_backward_mean = np.mean(A_sol_backward, axis=1)
    R_backward_mean = np.mean(R_sol_backward, axis=1)

    if plot:
        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("ARM Phase space plants")
        axs.plot(dAs, P_mean, color="blue")
        axs.plot(np.flip(dAs), P_backward_mean, color="red")
        axs.set_xlabel(r"Driver of decline")
        axs.set_ylabel(r"$\langle P \rangle$")
        if save_fig:
            plt.savefig(f"figures/state_space_P_G_{G}.png", format="png", dpi=500)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("ARM Phase space pollinators")
        axs.plot(dAs, A_mean, color="blue")
        axs.plot(np.flip(dAs), A_backward_mean, color="red")
        axs.set_xlabel(r"Driver of decline")
        axs.set_ylabel(r"$\langle A \rangle$")
        if save_fig:
            plt.savefig(f"figures/state_space_A_G_{G}.png", format="png", dpi=500)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("ARM Phase space resources")
        axs.plot(dAs, R_mean, color="blue")
        axs.plot(np.flip(dAs), R_backward_mean, color="red")
        axs.set_xlabel(r"Driver of decline")
        axs.set_ylabel(r"$\langle R \rangle$")
        if save_fig:
            plt.savefig(f"figures/state_space_R_G_{G}.png", format="png", dpi=500)

    if fname is not None:
        np.savez(
            fname, dAs=dAs, P_mean=P_mean, A_mean=A_mean, R_mean=R_mean,
            P_backward_mean=P_backward_mean, A_backward_mean=A_backward_mean,
            R_backward_mean=R_backward_mean
        )

    return (
        dAs, P_mean, A_mean, R_mean, P_backward_mean, A_backward_mean,
        R_backward_mean
    )


def state_space_BM(seed=None, plot=False, save_fig=False, fname=None):

    if seed is None:
        seed = np.random.SeedSequence()

    rng1 = np.random.default_rng(seed)

    N_p = 25
    N_a = 25
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_end = 500
    t_sol = 25

    dAs = np.linspace(0, 2.5, 40)

    # save only steady state solutions
    P_sol = np.zeros((len(dAs), N_p))
    A_sol = np.zeros((len(dAs), N_a))
    P_sol_backward = np.zeros((len(dAs), N_p))
    A_sol_backward = np.zeros((len(dAs), N_a))

    BM = pc.BaseModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, seed=seed,
        rng=rng1
    )

    # obtain initial solution
    sol = BM.solve(t_end, dA=0)

    # vis.plot_time_sol_pollcomm(sol, AM, dA=0)

    t0 = timer()
    for i, dA in enumerate(dAs):
        print(f"{i+1} out of {len(dAs)}...")

        y0 = sol.y[:, -1]

        sol = BM.solve(t_sol, dA=dA, y0=y0)

        P_sol[i] = sol.y[:BM.N_p, -1]
        A_sol[i] = sol.y[BM.N_p:BM.N, -1]

    print(f"Phase space forward. Time elapsed: {timer()-t0:.5f} seconds\n")

    t0 = timer()
    for i, dA in enumerate(np.flip(dAs)):
        print(f"{i+1} out of {len(dAs)}...")

        y0 = sol.y[:, -1]

        sol = BM.solve(t_sol, dA=dA, y0=y0)

        P_sol_backward[i] = sol.y[:BM.N_p, -1]
        A_sol_backward[i] = sol.y[BM.N_p:BM.N, -1]

    print(f"Phase space backward. Time elapsed: {timer()-t0:.5f} seconds\n")

    # calculate mean abundancies
    P_mean = np.mean(P_sol, axis=1)
    # P_std = np.std(P_sol, axis=1, ddof=1)
    A_mean = np.mean(A_sol, axis=1)
    # A_std = np.std(A_sol, axis=1, ddof=1)
    P_backward_mean = np.mean(P_sol_backward, axis=1)
    A_backward_mean = np.mean(A_sol_backward, axis=1)

    if plot:
        # fig, axs = plt.subplots(constrained_layout=True)
        # fig.suptitle(r"BM Phase space plants $d_A$")
        # axs.plot(dAs, P_mean, color="blue")
        # # axs.fill_between(dAs, P_mean-P_std, P_mean+P_std, alpha=0.3, color="blue")
        # axs.plot(np.flip(dAs), P_backward_mean, color="red")
        # axs.set_xlabel(r"Driver of decline")
        # axs.set_ylabel(r"$\langle P \rangle$")
        # if save_fig:
        #     plt.savefig(f"figures/state_space_P_BM.png", format="png", dpi=300)
        #
        # fig, axs = plt.subplots(constrained_layout=True)
        # fig.suptitle(r"BM Phase space pollinators $d_A$")
        # axs.plot(dAs, A_mean, color="blue")
        # axs.plot(np.flip(dAs), A_backward_mean, color="red")
        # axs.set_xlabel(r"Driver of decline")
        # axs.set_ylabel(r"$\langle A \rangle$")
        # if save_fig:
        #     plt.savefig(f"figures/BM_state_space_A_BM.png", format="png", dpi=300)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle(r"BM Phase space plants $d_A$")
        for j in range(P_sol.shape[1]):
            # axs.plot(dAs, P_sol[:, j])
            axs.plot(np.flip(dAs), P_sol_backward[:, j])
        # axs.fill_between(dAs, P_mean-P_std, P_mean+P_std, alpha=0.3, color="blue")

        axs.set_xlabel(r"Driver of decline")
        axs.set_ylabel(r"$\langle P \rangle$")
        if save_fig:
            plt.savefig(f"figures/state_space_P_BM.png", format="png", dpi=300)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle(r"BM Phase space pollinators $d_A$")
        for j in range(A_sol.shape[1]):
            # axs.plot(dAs, A_sol[:, j])
            axs.plot(np.flip(dAs), A_sol_backward[:, j])
        axs.set_xlabel(r"Driver of decline")
        axs.set_ylabel(r"$\langle A \rangle$")
        if save_fig:
            plt.savefig(f"figures/BM_state_space_A_BM.png", format="png", dpi=300)

    if fname is not None:
        np.savez(
            fname, dAs_BM=dAs, P_mean_BM=P_mean, A_mean_BM=A_mean,
            P_backward_mean_BM=P_backward_mean, A_backward_mean_BM=A_backward_mean
        )

    return dAs, P_mean, A_mean, P_backward_mean, A_backward_mean


def state_space_C_BM(seed=None, plot=False, save_fig=False, fname=None):

    if seed is None:
        seed = np.random.SeedSequence()

    rng1 = np.random.default_rng(seed)

    N_p = 20
    N_a = 20
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_end = 500
    t_sol = 150

    dCs = np.linspace(0.5, 10, 50)

    # save only steady state solutions
    P_sol = np.zeros((len(dCs), N_p))
    A_sol = np.zeros((len(dCs), N_a))
    P_sol_backward = np.zeros((len(dCs), N_p))
    A_sol_backward = np.zeros((len(dCs), N_a))

    BM = pc.BaseModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=rng1
    )

    # obtain initial solution
    sol = BM.solve(t_end)

    # vis.plot_time_sol_pollcomm(sol, AM, dA=0)

    t0 = timer()
    for i, dC in enumerate(dCs):
        print(f"{i+1} out of {len(dCs)}...")

        y0 = sol.y[:, -1]

        sol = BM.solve(t_sol, dC=dC, y0=y0)

        P_sol[i] = sol.y[:BM.N_p, -1]
        A_sol[i] = sol.y[BM.N_p:BM.N, -1]

    print(f"Phase space forward. Time elapsed: {timer()-t0:.5f} seconds\n")

    t0 = timer()
    for i, dC in enumerate(np.flip(dCs)):
        print(f"{i+1} out of {len(dCs)}...")

        y0 = sol.y[:, -1]

        sol = BM.solve(t_sol, dC=dC, y0=y0)

        P_sol_backward[i] = sol.y[:BM.N_p, -1]
        A_sol_backward[i] = sol.y[BM.N_p:BM.N, -1]

    print(f"Phase space backward. Time elapsed: {timer()-t0:.5f} seconds\n")

    # calculate mean abundancies
    P_mean = np.mean(P_sol, axis=1)
    # P_std = np.std(P_sol, axis=1, ddof=1)
    A_mean = np.mean(A_sol, axis=1)
    # A_std = np.std(A_sol, axis=1, ddof=1)
    P_backward_mean = np.mean(P_sol_backward, axis=1)
    A_backward_mean = np.mean(A_sol_backward, axis=1)

    if plot:
        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("BM State space plants competition")
        axs.plot(dCs, P_mean, color="blue")
        axs.plot(np.flip(dCs), P_backward_mean, color="red")
        axs.set_xlabel(r"Relative change in competition")
        axs.set_ylabel(r"$\langle P \rangle$")
        # axs.set_xscale("log")
        # axs.set_yscale("log")
        if save_fig:
            plt.savefig(f"figures/state_space_P_BM_C.png", format="png", dpi=500)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("BM State space pollinators competition")
        axs.plot(dCs, A_mean, color="blue")
        axs.plot(np.flip(dCs), A_backward_mean, color="red")
        axs.set_xlabel(r"Relative change in competition")
        axs.set_ylabel(r"$\langle A \rangle$")
        # axs.set_xscale("log")
        # axs.set_yscale("log")
        if save_fig:
            plt.savefig(f"figures/state_space_A_BM_C.png", format="png", dpi=500)

    if fname is not None:
        np.savez(
            fname, dCs_BM=dCs, P_mean_BM=P_mean, A_mean_BM=A_mean,
            P_backward_mean_BM=P_backward_mean, A_backward_mean_BM=A_backward_mean
        )

    return dCs, P_mean, A_mean, P_backward_mean, A_backward_mean


def state_space_VM(seed=None, plot=False, save_fig=False, fname=None):

    if seed is None:
        seed = np.random.SeedSequence()

    rng1 = np.random.default_rng(seed)

    N_p = 20
    N_a = 20
    mu = 0.001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_end = 5000
    t_sol = 5000

    dAs = np.linspace(0, 0.16, 50)

    # save only steady state solutions
    P_sol = np.zeros((len(dAs), N_p))
    A_sol = np.zeros((len(dAs), N_a))
    R_sol = np.zeros((len(dAs), N_p))
    P_sol_backward = np.zeros((len(dAs), N_p))
    A_sol_backward = np.zeros((len(dAs), N_a))
    R_sol_backward = np.zeros((len(dAs), N_p))

    VM = pc.ValdovinosModel (
        N_p, N_a, mu, connectance, forbidden, nestedness, rng=rng1, seed=seed
    )

    # obtain initial solution
    sol = VM.solve(t_end, dA=0, save_period=0)

    t0 = timer()
    for i, dA in enumerate(dAs):
        print(f"{i+1} out of {len(dAs)}...")

        y0 = sol.y[:, -1]

        y0 = np.concatenate((y0, sol.y_partial[:, -1]))

        sol = VM.solve(t_sol, dA=dA, y0=y0, save_period=0)

        P_sol[i] = sol.y[:VM.N_p, -1]
        A_sol[i] = sol.y[VM.N_p:VM.N, -1]
        R_sol[i] = sol.y[VM.N:, -1]

    print(f"Phase space forward. Time elapsed: {timer()-t0:.5f} seconds\n")

    t0 = timer()
    for i, dA in enumerate(np.flip(dAs)):
        print(f"{i+1} out of {len(dAs)}...")

        y0 = sol.y[:, -1]
        y0 = np.concatenate((y0, sol.y_partial[:, -1]))

        sol = VM.solve(t_sol   , dA=dA, y0=y0, save_period=0)

        P_sol_backward[i] = sol.y[:VM.N_p, -1]
        A_sol_backward[i] = sol.y[VM.N_p:VM.N, -1]
        R_sol_backward[i] = sol.y[VM.N:, -1]

    print(f"Phase space backward. Time elapsed: {timer()-t0:.5f} seconds\n")

    # calculate mean abundancies
    P_mean = np.mean(P_sol, axis=1)
    A_mean = np.mean(A_sol, axis=1)
    R_mean = np.mean(R_sol, axis=1)
    P_backward_mean = np.mean(P_sol_backward, axis=1)
    A_backward_mean = np.mean(A_sol_backward, axis=1)
    R_backward_mean = np.mean(R_sol_backward, axis=1)

    if plot:
        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("VM Phase space plants")
        axs.plot(dAs, P_mean, color="blue")
        axs.plot(np.flip(dAs), P_backward_mean, color="red")
        axs.set_xlabel(r"Driver of decline")
        axs.set_ylabel(r"$\langle P \rangle$")
        if save_fig:
            plt.savefig("figures/VM_state_space_P.png", format="png", dpi=500)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("VM Phase space pollinators")
        axs.plot(dAs, A_mean, color="blue")
        axs.plot(np.flip(dAs), A_backward_mean, color="red")
        axs.set_xlabel(r"Driver of decline")
        axs.set_ylabel(r"$\langle A \rangle$")
        if save_fig:
            plt.savefig("figures/VM_state_space_A.png", format="png", dpi=500)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("VM Phase space resources")
        axs.plot(dAs, R_mean, color="blue")
        axs.plot(np.flip(dAs), R_backward_mean, color="red")
        axs.set_xlabel(r"Driver of decline")
        axs.set_ylabel(r"$\langle R \rangle$")
        if save_fig:
            plt.savefig("figures/VM_state_space_R.png", format="png", dpi=500)

    if fname is not None:
        np.savez(
            fname, dAs=dAs, P_mean=P_mean, A_mean=A_mean, R_mean=R_mean,
            P_backward_mean=P_backward_mean, A_backward_mean=A_backward_mean,
            R_backward_mean=R_backward_mean
        )

    return (
        dAs, P_mean, A_mean, R_mean, P_backward_mean, A_backward_mean,
        R_backward_mean
    )


def state_space_rate_AM(G=1, seed=None, plot=False, save_fig=False, fname=None):

    if seed is None:
        seed = np.random.SeedSequence()

    rng1 = np.random.default_rng(seed)

    N_p = 20
    N_a = 20
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_init = 500
    t_const = 0

    AM = pc.AdaptiveModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=rng1, G=G
    )

    # obtain initial steady-state solution
    sol = AM.solve(t_init, dA=0)
    y0 = sol.y[:, -1]
    y0 = np.concatenate((y0, sol.y_partial[:, -1]))

    rates = np.linspace(0.01, 4, 4)
    def dA_rate(t, r, dA_max):
        t_max = dA_max / r
        if t <= t_max:
            return r * t
        elif t > t_max:
            return dA_max
    dA = {
        "func": dA_rate,
        "args": None
    }

    dA_maxs = [0.1, 1, 1.5, 4]
    P_means = np.zeros((len(dA_maxs), len(rates)))
    A_means = np.zeros((len(dA_maxs), len(rates)))

    t0 = timer()
    for j, dA_max in enumerate(dA_maxs):
        print(f"{j+1} out of {len(dA_maxs)}...")
        dAs = []
        for r in rates:
            dA_copy = copy.copy(dA)
            dA_copy["args"] = (r, dA_max)
            dAs.append(dA_copy)

        # save only steady state solutions
        P_sol = np.zeros((len(rates), N_p))
        A_sol = np.zeros((len(rates), N_a))

        for i, dA in enumerate(dAs):

            t_end = (dA_max / dA["args"][0]) + t_const
            sol = AM.solve(t_end, dA=dA, y0=y0, save_period=0)

            # vis.plot_time_sol_pollcomm(sol, AM, dA=0)
            P_sol[i] = sol.y[:AM.N_p, -1]
            A_sol[i] = sol.y[AM.N_p:AM.N, -1]


        # calculate mean abundancies
        P_means[j] = np.mean(P_sol, axis=1)
        A_means[j] = np.mean(A_sol, axis=1)

    print(f"Phase space. Time elapsed: {timer()-t0:.5f} seconds\n")

    if plot:
        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("AM Phase space plants")
        for i, P_mean in enumerate(P_means):
            axs.plot(rates, P_mean, label=f"dA_max = {dA_maxs[i]}")
        axs.set_xlabel(r"Rate")
        axs.set_ylabel(r"$\langle P \rangle$")
        # axs.set_ylim(0, 2)
        axs.legend()
        if save_fig:
            plt.savefig(f"figures/state_space_rate_AM.png", format="png", dpi=300)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("AM Phase space pollinators")
        for i, A_mean in enumerate(A_means):
            axs.plot(rates, A_mean, label=f"dA_max = {dA_maxs[i]}")
        axs.set_xlabel(r"Rate")
        axs.set_ylabel(r"$\langle A \rangle$")
        # axs.set_ylim(0, 2)
        axs.legend()
        if save_fig:
            plt.savefig(f"figures/state_space_rate_AM.png", format="png", dpi=300)

    if fname is not None:
        np.savez(
            fname, dA_maxs=dA_maxs, rates=rates, P_means=P_means, A_means=A_means
        )

    return dA_maxs, rates, P_means, A_means


def state_space_rate_ARM(G=1, seed=None, plot=False, save_fig=False, fname=None):

    if seed is None:
        seed = np.random.SeedSequence()

    rng1 = np.random.default_rng(seed)

    N_p = 15
    N_a = 35
    mu = 0.001
    connectance = 0.2
    forbidden = 0.25
    nestedness = 0.6
    network_type = "nested"
    t_const = 50


    ARM = pc.AdaptiveExplicitResourceModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=rng1,
        seed=seed, G=G
    )

    # obtain initial steady-state solution
    sol = ARM.solve(200, dA=0)
    y0 = sol.y[:, -1]
    y0 = np.concatenate((y0[:ARM.N], sol.y_partial[:, -1], y0[ARM.N:]))

    rates = np.linspace(0.01, 0.75, 20)
    def dA_rate(t, r, dA_max):
        t_max = dA_max / r
        if t <= t_max:
            return r * t
        elif t > t_max:
            return dA_max
    dA = {
        "func": dA_rate,
        "args": None
    }

    dA_maxs = [0, 0.5, 1, 1.5, 3]
    P_means = np.zeros((len(dA_maxs), len(rates)))
    A_means = np.zeros((len(dA_maxs), len(rates)))
    R_means = np.zeros((len(dA_maxs), len(rates)))

    t0 = timer()
    for j, dA_max in enumerate(dA_maxs):
        print(f"{j+1} out of {len(dA_maxs)}...")
        dAs = []
        for r in rates:
            dA_copy = copy.copy(dA)
            dA_copy["args"] = (r, dA_max)
            dAs.append(dA_copy)

        # save only steady state solutions
        P_sol = np.zeros((len(rates), N_p))
        A_sol = np.zeros((len(rates), N_a))
        R_sol = np.zeros((len(rates), N_p))

        for i, dA in enumerate(dAs):

            t_end = (dA_max / dA["args"][0]) + t_const
            sol = ARM.solve(t_end, dA=dA, y0=y0, save_period=0)

            # vis.plot_time_sol_pollcomm(sol, ARM, dA=0)
            P_sol[i] = sol.y[:ARM.N_p, -1]
            A_sol[i] = sol.y[ARM.N_p:ARM.N, -1]
            R_sol[i] = sol.y[ARM.N:, -1]

        # calculate mean abundancies
        P_means[j] = np.mean(P_sol, axis=1)
        A_means[j] = np.mean(A_sol, axis=1)
        R_means[j] = np.mean(R_sol, axis=1)

    print(f"Phase space rate. Time elapsed: {timer()-t0:.5f} seconds\n")

    if plot:
        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("ARM Phase space plants")
        for i, P_mean in enumerate(P_means):
            axs.plot(rates, P_mean, label=f"dA_max = {dA_maxs[i]}")
        axs.set_xlabel(r"Rate")
        axs.set_ylabel(r"$\langle P \rangle$")
        # axs.set_ylim(0, 2)
        axs.legend()
        if save_fig:
            plt.savefig(f"figures/state_space_rate_ARM_plants.png", format="png", dpi=300)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("ARM Phase space pollinators")
        for i, A_mean in enumerate(A_means):
            axs.plot(rates, A_mean, label=f"dA_max = {dA_maxs[i]}")
        axs.set_xlabel(r"Rate")
        axs.set_ylabel(r"$\langle A \rangle$")
        # axs.set_ylim(0, 2)
        axs.legend()
        if save_fig:
            plt.savefig(f"figures/state_space_rate_ARM_polls.png", format="png", dpi=300)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("ARM Phase space resources")
        for i, R_mean in enumerate(R_means):
            axs.plot(rates, R_mean, label=f"dA_max = {dA_maxs[i]}")
        axs.set_xlabel(r"Rate")
        axs.set_ylabel(r"$\langle R \rangle$")
        # axs.set_ylim(0, 2)
        axs.legend()
        if save_fig:
            plt.savefig(f"figures/state_space_rate_ARM_resources.png", format="png", dpi=500)

    if fname is not None:
        np.savez(
            fname, dA_maxs=dA_maxs, rates=rates, P_means=P_means, A_means=A_means,
            R_means=R_means
        )

        return dA_maxs, rates, P_means, A_means, R_means


def state_space_rate_BM(seed=None, plot=False, save_fig=False, fname=None):

    if seed is None:
        seed = np.random.SeedSequence()

    rng1 = np.random.default_rng(seed)

    N_p = 25
    N_a = 25
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_init = 500
    t_const = 100

    BM = pc.BaseModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=rng1
    )

    # obtain initial steady-state solution
    sol = BM.solve(t_init, dA=0)
    y0 = sol.y[:, -1]

    rates = np.linspace(0.0001, 10, 15)
    def dA_rate(t, r, dA_max):
        t_max = dA_max / r
        if t <= t_max:
            return r * t
        elif t > t_max:
            return dA_max
    dA = {
        "func": dA_rate,
        "args": None
    }

    dA_maxs = [0.1, 1, 1.5, 4]
    P_means = np.zeros((len(dA_maxs), len(rates)))
    A_means = np.zeros((len(dA_maxs), len(rates)))

    t0 = timer()
    for j, dA_max in enumerate(dA_maxs):
        print(f"{j+1} out of {len(dA_maxs)}...")
        dAs = []
        for r in rates:
            dA_copy = copy.copy(dA)
            dA_copy["args"] = (r, dA_max)
            dAs.append(dA_copy)

        # save only steady state solutions
        P_sol = np.zeros((len(rates), N_p))
        A_sol = np.zeros((len(rates), N_a))

        for i, dA in enumerate(dAs):

            t_end = (dA_max / dA["args"][0]) + t_const
            sol = BM.solve(t_end, dA=dA, y0=y0)

            # vis.plot_time_sol_pollcomm(sol, AM, dA=0)
            P_sol[i] = sol.y[:BM.N_p, -1]
            A_sol[i] = sol.y[BM.N_p:BM.N, -1]


        # calculate mean abundancies
        P_means[j] = np.mean(P_sol, axis=1)
        A_means[j] = np.mean(A_sol, axis=1)

    print(f"Phase space. Time elapsed: {timer()-t0:.5f} seconds\n")

    if plot:
        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("BM Phase space plants")
        for i, P_mean in enumerate(P_means):
            axs.plot(rates, P_mean, label=f"dA_max = {dA_maxs[i]}")
        axs.set_xlabel(r"Rate")
        axs.set_ylabel(r"$\langle P \rangle$")
        # axs.set_ylim(0, 2)
        axs.legend()
        if save_fig:
            plt.savefig(f"figures/state_space_rate_BM.png", format="png", dpi=300)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("BM Phase space pollinators")
        for i, A_mean in enumerate(A_means):
            axs.plot(rates, A_mean, label=f"dA_max = {dA_maxs[i]}")
        axs.set_xlabel(r"Rate")
        axs.set_ylabel(r"$\langle A \rangle$")
        # axs.set_ylim(0, 2)
        axs.legend()
        if save_fig:
            plt.savefig(f"figures/state_space_rate_BM.png", format="png", dpi=300)

    if fname is not None:
        np.savez(
            fname, dA_maxs=dA_maxs, rates=rates, P_means=P_means, A_means=A_means
        )

    return dA_maxs, rates, P_means, A_means


def state_space_rate_VM(seed=None, plot=False, save_fig=False, fname=None):

    if seed is None:
        seed = np.random.SeedSequence()

    rng1 = np.random.default_rng(seed)

    N_p = 20
    N_a = 20
    mu = 0.001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_init = 5000
    t_const = 0

    VM = pc.ValdovinosModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, rng=rng1, seed=seed
    )

    # obtain initial steady-state solution
    sol = VM.solve(t_init, dA=0)
    y0 = sol.y[:, -1]
    y0 = np.concatenate((y0, sol.y_partial[:, -1]))

    rates = np.linspace(0.0001, 0.001, 4)
    def dA_rate(t, r, dA_max):
        t_max = dA_max / r
        if t <= t_max:
            return r * t
        elif t > t_max:
            return dA_max
    dA = {
        "func": dA_rate,
        "args": None
    }

    dA_maxs = [0.0001, 0.01, 0.05, 0.8]
    P_means = np.zeros((len(dA_maxs), len(rates)))
    A_means = np.zeros((len(dA_maxs), len(rates)))
    R_means = np.zeros((len(dA_maxs), len(rates)))

    t0 = timer()
    for j, dA_max in enumerate(dA_maxs):
        print(f"{j+1} out of {len(dA_maxs)}...")
        dAs = []
        for r in rates:
            dA_copy = copy.copy(dA)
            dA_copy["args"] = (r, dA_max)
            dAs.append(dA_copy)

        # save only steady state solutions
        P_sol = np.zeros((len(rates), N_p))
        A_sol = np.zeros((len(rates), N_a))
        R_sol = np.zeros((len(rates), N_p))

        for i, dA in enumerate(dAs):

            t_end = (dA_max / dA["args"][0]) + t_const
            sol = VM.solve(t_end, dA=dA, y0=y0, save_period=0)

            # vis.plot_time_sol_pollcomm(sol, ARM, dA=0)
            P_sol[i] = sol.y[:VM.N_p, -1]
            A_sol[i] = sol.y[VM.N_p:VM.N, -1]
            R_sol[i] = sol.y[VM.N:, -1]

        # calculate mean abundancies
        P_means[j] = np.mean(P_sol, axis=1)
        A_means[j] = np.mean(A_sol, axis=1)
        R_means[j] = np.mean(R_sol, axis=1)

    print(f"Phase space rate. Time elapsed: {timer()-t0:.5f} seconds\n")

    if plot:
        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("VM Phase space plants")
        for i, P_mean in enumerate(P_means):
            axs.plot(rates, P_mean, label=f"dA_max = {dA_maxs[i]}")
        axs.set_xlabel(r"Rate")
        axs.set_ylabel(r"$\langle P \rangle$")
        # axs.set_ylim(0, 2)
        axs.legend()
        if save_fig:
            plt.savefig("figures/state_space_rate_VM_plants.png", format="png", dpi=500)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("VM Phase space pollinators")
        for i, A_mean in enumerate(A_means):
            axs.plot(rates, A_mean, label=f"dA_max = {dA_maxs[i]}")
        axs.set_xlabel(r"Rate")
        axs.set_ylabel(r"$\langle A \rangle$")
        # axs.set_ylim(0, 2)
        axs.legend()
        if save_fig:
            plt.savefig("figures/state_space_rate_VM_polls.png", format="png", dpi=500)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("VM Phase space resources")
        for i, R_mean in enumerate(R_means):
            axs.plot(rates, R_mean, label=f"dA_max = {dA_maxs[i]}")
        axs.set_xlabel(r"Rate")
        axs.set_ylabel(r"$\langle R \rangle$")
        # axs.set_ylim(0, 2)
        axs.legend()
        if save_fig:
            plt.savefig("figures/state_space_rate_VM_resources.png", format="png", dpi=500)

    if fname is not None:
        np.savez(
            fname, dA_maxs=dA_maxs, rates=rates, P_means=P_means, A_means=A_means,
            R_means=R_means
        )

    return dA_maxs, rates, P_means, A_means, R_means


def state_space_number_species_AM(seed=None, plot=False, save_fig=False):

    if seed is None:
        seed = np.random.SeedSequence()

    rng = np.random.default_rng(seed)

    N_p = 25
    N_a = 25
    mu = 0.0001
    nu = 0.1
    q = 1
    G = 1
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_init = 500
    t_const = 0

    model = AdaptiveModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=rng1, G=G,
        q=q, nu=nu
    )

    # obtain initial steady-state solution
    model.solve(t_init, dA=0)
    y0 = model.y[:, -1]
    y0 = np.concatenate((y0, model.y_partial[:, -1]))

    rates = np.linspace(0.01, 4, 4)
    def dA_rate(t, r, dA_max):
        t_max = dA_max / r
        if t <= t_max:
            return r * t
        elif t > t_max:
            return dA_max
    dA = {
        "func": dA_rate,
        "args": None
    }

    dA_maxs = [0.1, 1, 1.5, 4]
    P_means = np.zeros((len(dA_maxs), len(rates)))
    A_means = np.zeros((len(dA_maxs), len(rates)))

    t0 = timer()
    for j, dA_max in enumerate(dA_maxs):
        print(f"{j+1} out of {len(dA_maxs)}...")
        dAs = []
        for r in rates:
            dA_copy = copy.copy(dA)
            dA_copy["args"] = (r, dA_max)
            dAs.append(dA_copy)

        # save only steady state solutions
        P_sol = np.zeros((len(rates), N_p))
        A_sol = np.zeros((len(rates), N_a))

        for i, dA in enumerate(dAs):

            t_end = (dA_max / dA["args"][0]) + t_const
            sol = AM.solve(t_end, dA=dA, y0=y0, save_period=0)

            # vis.plot_time_sol_pollcomm(sol, AM, dA=0)
            P_sol[i] = sol.y[:AM.N_p, -1]
            A_sol[i] = sol.y[AM.N_p:AM.N, -1]


        # calculate mean abundancies
        P_means[j] = np.mean(P_sol, axis=1)
        A_means[j] = np.mean(A_sol, axis=1)

    print(f"Phase space. Time elapsed: {timer()-t0:.5f} seconds\n")

    if plot:
        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("AM Phase space plants")
        for i, P_mean in enumerate(P_means):
            axs.plot(rates, P_mean, label=f"dA_max = {dA_maxs[i]}")
        axs.set_xlabel(r"Rate")
        axs.set_ylabel(r"$\langle P \rangle$")
        # axs.set_ylim(0, 2)
        axs.legend()
        if save_fig:
            plt.savefig(f"figures/state_space_rate_AM.png", format="png", dpi=300)

        fig, axs = plt.subplots(constrained_layout=True)
        fig.suptitle("AM Phase space pollinators")
        for i, A_mean in enumerate(A_means):
            axs.plot(rates, A_mean, label=f"dA_max = {dA_maxs[i]}")
        axs.set_xlabel(r"Rate")
        axs.set_ylabel(r"$\langle A \rangle$")
        # axs.set_ylim(0, 2)
        axs.legend()
        if save_fig:
            plt.savefig(f"figures/state_space_rate_AM.png", format="png", dpi=300)

    if fname is not None:
        np.savez(
            fname, dA_maxs=dA_maxs, rates=rates, P_means=P_means, A_means=A_means
        )

    return dA_maxs, rates, P_means, A_means


def state_space_abundance_rate_AM(
    seed=None, plot=False, save_fig=False, fname=None, recalculate=True, q=1
):
    t0 = timer()

    if seed is None:
        seed = np.random.SeedSequence()

    rng = np.random.default_rng(seed)

    N_p = 25
    N_a = 25
    mu = 0.0001
    nu = 0.5
    G = 0.2
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_end = 500     # max. simulation time
    n_steps = int(1e5) # number of interpolated time steps
    extinct_threshold = 0.01    # abundance at which a population is defined as extinct

    def dA_rate(t, r):
        return r * t

    pollcomm = pc.AdaptiveModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=rng, G=G,
        q=q, nu=nu, feasible=True, feasible_iters=50
    )
    if fname is None:
        fname = f"output/state_space_abundance_rate_{repr(pollcomm)}_nu{nu}_q{q}_G{G}"

    if recalculate:
        rate_init = np.linspace(0.001, 0.25, 15)
        abundance_init = np.linspace(0, 1, 15)

        dA_critical = np.zeros((len(rate_init), len(abundance_init)))

        curr_iter = 0
        total_iter = len(rate_init) * len(abundance_init)
        for i, rate in enumerate(rate_init):
            for j, abundance in enumerate(abundance_init):
                print(f"Iteration {curr_iter + 1} out of {total_iter}")

                # initial conditions
                y0 = np.full(pollcomm.N, abundance, dtype=float)
                y0 = np.concatenate((y0, copy.deepcopy(pollcomm.alpha.flatten())))

                # drivers of decline
                dA = {
                    "func": dA_rate,
                    "args": (rate, )
                }

                sol = pollcomm.solve(
                    t_end, dA=dA, n_steps=n_steps, y0=y0, stop_on_collapse=True,
                    extinct_threshold=extinct_threshold
                )

                # check if point of collapse has been found:
                if sol.status == 1:
                    # find point of collapse
                    A_mean = pollcomm.y[pollcomm.N_p:pollcomm.N].mean(axis=0)

                    # put default at -1 if no population went extinct
                    try:
                        ind = (A_mean < extinct_threshold).nonzero()[0][0]
                        t_extinct = pollcomm.t[ind]
                        dA_critical[i, j] = dA["func"](t_extinct, rate)
                    except IndexError:
                        dA_critical[i, j] = -1
                else:
                    dA_critical[i, j] = -1

                curr_iter += 1

        np.savez(
            fname, rate_init=rate_init, abundance_init=abundance_init,
            dA_critical=dA_critical,
        )

    print(f"\nTotal simulation time: {timer()-t0:.2f} seconds")

    if plot:
        vis.plot_state_space_abundance_rate(pollcomm, fname+".npz", save_fig=save_fig)


def state_space_abundance_env_AM(
    seed=None, plot=False, save_fig=False, fname=None, recalculate=True, q=0.2,
    G=0.2, nu=0.5
):

    t0 = timer()

    if seed is None:
        seed = np.random.SeedSequence()

    rng = np.random.default_rng(seed)

    N_p = 25
    N_a = 25
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_end = 500     # max. simulation time
    n_steps = int(1e4) # number of interpolated time steps

    # abundance at which a population is defined as extinct
    # if None, plot based on final abundance level instead of extinction
    extinct_threshold = None

    pollcomm = pc.AdaptiveModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=rng, G=G,
        q=q, nu=nu, feasible=True, feasible_iters=50
    )

    if fname is None:
        fname = f"output/state_space_abundance_env_{repr(pollcomm)}_nu{nu}_q{q}_G{G}"

    if recalculate:
        dA_init = np.linspace(0.4, 1.5, 16)
        abundance_init = np.linspace(0, 0.6, 16)

        final_abundance = np.zeros((len(dA_init), len(abundance_init)))

        curr_iter = 0
        total_iter = len(dA_init) * len(abundance_init)

        for i, dA in enumerate(dA_init):
            for j, abundance in enumerate(abundance_init):
                print(f"Iteration {curr_iter + 1} out of {total_iter}")

                # initial conditions
                y0 = np.full(pollcomm.N, abundance, dtype=float)
                y0 = np.concatenate((y0, copy.deepcopy(pollcomm.alpha.flatten())))

                sol = pollcomm.solve(
                    t_end, dA=dA, n_steps=n_steps, y0=y0, stop_on_collapse=False,
                    stop_on_equilibrium=True
                )

                A_mean = pollcomm.y[pollcomm.N_p:pollcomm.N].mean(axis=0)

                # final abundace is the mean abundace at the final time point
                final_abundance[i, j] = A_mean[-1]

                curr_iter += 1

        np.savez(
            fname, dA_init=dA_init, abundance_init=abundance_init,
            final_abundance=final_abundance,
        )

    print(f"\nTotal simulation time: {timer()-t0:.2f} seconds")

    if plot:
        vis.plot_state_space_abundance_env(
            pollcomm, fname+".npz", save_fig=save_fig, extinct_threshold=extinct_threshold
        )


def state_space_abundance_env_rate_AM(
    seed=None, plot=False, save_fig=False, fname=None, recalculate=True, q=0
):

    t0 = timer()

    if seed is None:
        seed = np.random.SeedSequence()

    rng = np.random.default_rng(seed)

    N_p = 25
    N_a = 25
    mu = 0.0001
    nu = 0.5
    G = 0
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_end = 1000     # max. simulation time
    n_steps = int(5e4) # number of interpolated time steps

    # abundance at which a population is defined as extinct
    extinct_threshold = 0.01

    def dA_rate(t, r, dA_0):
        return dA_0 + r * t

    pollcomm = pc.AdaptiveModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=rng, G=G,
        q=q, nu=nu, feasible=True, feasible_iters=50
    )

    if fname is None:
        fname = f"output/state_space_abundance_env_rate_{repr(pollcomm)}_nu{nu}_q{q}_G{G}"

    if recalculate:
        dA_init = np.linspace(0, 1.25, 10)
        abundance_init = np.array([0, 0.2, 0.7, 1.5])
        rate_init = np.linspace(0.0001, 0.2, 20)

        dA_critical = np.zeros((len(dA_init), len(abundance_init), len(rate_init)))

        curr_iter = 0
        total_iter = len(dA_init) * len(abundance_init) * len(rate_init)

        for i, dA in enumerate(dA_init):
            for j, abundance in enumerate(abundance_init):
                for k, rate in enumerate(rate_init):
                    print(f"Iteration {curr_iter + 1} out of {total_iter}")

                    # initial conditions
                    y0 = np.full(pollcomm.N, abundance, dtype=float)
                    y0 = np.concatenate((y0, copy.deepcopy(pollcomm.alpha.flatten())))

                    # driver of decline
                    dA_dict = {
                        "func": dA_rate,
                        "args": (rate, dA)
                    }

                    sol = pollcomm.solve(
                        t_end, dA=dA_dict, n_steps=n_steps, y0=y0, stop_on_collapse=True,
                        extinct_threshold=extinct_threshold
                    )

                    # check if point of collapse has been found:
                    if sol.status == 1:
                        # find point of collapse
                        A_mean = pollcomm.y[pollcomm.N_p:pollcomm.N].mean(axis=0)

                        # put default at -1 if no population went extinct
                        try:
                            ind = (A_mean < extinct_threshold).nonzero()[0][0]
                            t_extinct = pollcomm.t[ind]
                            dA_critical[i, j, k] = dA_dict["func"](t_extinct, rate, dA)
                        except IndexError:
                            dA_critical[i, j, k] = -1
                    else:
                        dA_critical[i, j, k] = -1

                    curr_iter += 1

        np.savez(
            fname, dA_init=dA_init, abundance_init=abundance_init, rate_init=rate_init,
            dA_critical=dA_critical
        )

    print(f"\nTotal simulation time: {timer()-t0:.2f} seconds")

    if plot:
        vis.plot_state_space_abundance_env_rate(
            pollcomm, fname+".npz", save_fig=save_fig, extinct_threshold=extinct_threshold
        )
