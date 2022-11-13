#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 13/05/2022
# ---------------------------------------------------------------------------
""" sa.py

Sensitivity Analysis

1) Sensitivity analysis on the feasibility of networks
2) Senstivity analysis on the point of collapse
"""
# ---------------------------------------------------------------------------
import copy
import datetime
import os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from tqdm import tqdm

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
plt.rc("font", family="serif", size=16.)
plt.rc("savefig", dpi=600)
plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
plt.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)
plt.rc("axes.spines", top=False, right=False)

# define as global variables, because it is difficult to pass it as argument
# N_REPS is the number repetitions for each parameter set
N_REPS = 10


def main():

    t0 = timer()

    print(f"\nStarted sensitivity analysis at {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    fnumber = 6
    # run_sa(512, fnumber, parallel=True)
    analyze_sa(fnumber)

    fnumber = 0
    # run_sa_dA_Sinit(512, fnumber, parallel=True)
    # return
    analyze_sa_dA_Sinit(fnumber)

    print(f"\nFinished sensitivity analysis at {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    print(f"Total time sensitivity analysis: {timer()-t0:.2f} seconds")

    return plt.show()


def run_sa(N, fnumber=0, parallel=False):
    """Perform the sensitivity analysis on the feasibility of networks
    N_REPS is number of times a single sampled parameter set is simulated.

    Parameters
    ----------
    N : int
        The number of samples per parameter. Total samples is N(2D + 2) with D the number
        of parameters varied.
    fnumber : int, optional
        The filenumber in which the results are saved
    parallelel : bool, optional
        If True, the sensitivity analysis is done parallel on all the cores available
    """
    # define problem
    problem = {
        "num_vars": 5,
        "names": ["q", "nestedness", "connectance", "nu", "mu"],
        "bounds": [
            [0, 1],
            [0.2, 0.7],
            [0.15, 0.35],
            [0, 1],
            [0.000001, 0.0001]
        ]
    }

    # generate parameter samples
    # in total N(2D + 2) samples (D is the amount of params, N samples per param)
    param_values = saltelli.sample(problem, N)

    n_params = param_values.shape[0]
    print(f"\nCalculating a total of {n_params} samples...")
    print("Expected runtime (estimating with 1.2 seconds per simulation):")
    print(f"\t{n_params*N_REPS*1.2:.2f} seconds")
    print(f"\t{n_params*N_REPS*1.2/(60):.2f} minutes")
    print(f"\t{n_params*N_REPS*1.2/(60*60):.2f} hours")

    # Parallel is faster as long as model simulation time is on average around
    # 0.1 seconds or more
    if parallel:
        # run model for each parameter set in parallel
        with mp.Pool(os.cpu_count()) as pool:
            results = np.asarray(pool.map(evaluate_model, param_values))

    else:
        # run model for each parameter set
        results = np.zeros((n_params, 4))
        for i in range(param_values.shape[0]):
            print(f"{i+1} out of {param_values.shape[0]}")
            params = param_values[i]
            results[i] = evaluate_model(params)

    # split results into four, since we keep track of plant species alive,
    # pollinator species alive, plant abundance, and pollinator abundance.
    P_alive = np.array([result[0] for result in results])
    A_alive = np.array([result[1] for result in results])
    P_abundance = np.array([result[2] for result in results])
    A_abundance = np.array([result[3] for result in results])

    # save results for later analysis
    fname = f"output/sa/sa_AM_{fnumber}.npz"
    np.savez(
        fname, P_alive=P_alive, A_alive=A_alive, P_abundance=P_abundance,
        A_abundance=A_abundance, problem_keys=np.asarray(list(problem.keys())),
        problem_items=np.asarray(list(problem.values()), dtype=object),
        n_reps=np.asarray(N_REPS)
    )


def evaluate_model(params):
    """Evaluates model for given parameters to calculate feasibility of species

    Parameters
    ----------
    params : array
        Containt the values of q, nestedness, connectance, nu, mu to evaluate the model
    Returns
    ---------
    P_alive : float
        Number of plants alive averaged over N_REPS repetitions
    A_alive : float
        Number of pollinators alive averaged over N_REPS repetitions
    P_abundance : float
        Plant abundance averaged over N_REPS repetitions
    A_abundance : float
        Pollinator abundance averaged over N_REPS repetitions
    """
    q, nestedness, connectance, nu, mu = params

    # maximum simulation time, even if equilibrium is not reached
    t_end = int(1e4)
    n_steps = int(1e4)
    N_p = 15
    N_a = 35

    P_alive_temp = np.zeros(N_REPS)
    A_alive_temp = np.zeros(N_REPS)
    P_temp = np.zeros(N_REPS)
    A_temp = np.zeros(N_REPS)
    for rep in range(N_REPS):

        AM = pc.AdaptiveModel(
            N_p, N_a, mu=mu, connectance=connectance, forbidden=0.3,
            nestedness=nestedness, network_type="nested", rng=None, seed=None, nu=nu,
            G=1, q=q, feasible=False, beta_trade_off=0.5
        )

        # we need to solve at enough time points (n_steps), because at each time step,
        # we check if we are at equilibrium. However, we only need to store solution at
        # last time point, so choose n_steps not too large to avoid redundant computation
        AM.solve(
            t_end, dA=0, n_steps=n_steps, stop_on_equilibrium=True, save_period=None
        )

        P_alive_temp[rep] = AM.y_alive[:AM.N_p, -1].sum()
        A_alive_temp[rep] = AM.y_alive[AM.N_p:, -1].sum()
        P_temp[rep] = AM.y[:AM.N_p, -1].mean()
        A_temp[rep] = AM.y[AM.N_p:, -1].mean()

    # return number of plant species alive, number of pollinator species alive,
    # plant abundance, and pollinator abundance at the end of the simulation
    P_alive = np.mean(P_alive_temp, axis=0)
    A_alive = np.mean(A_alive_temp, axis=0)
    P_abundance = np.mean(P_temp, axis=0)
    A_abundance = np.mean(A_temp, axis=0)

    return P_alive, A_alive, P_abundance, A_abundance


def analyze_sa(fnumber):
    """Calculate the sobol indices for the sensitivity analysis on the feasibility
    of the networks and plot them.

    Parameters
    ----------
    fnumber : int
        The filenumber in which the results are saved
    """

    # load results of sa
    fname = f"output/sa/sa_AM_{fnumber}.npz"
    results = np.load(fname, allow_pickle=True)
    P_alive = results["P_alive"]
    A_alive = results["A_alive"]
    P_abundance = results["P_abundance"]
    A_abundance = results["A_abundance"]
    problem_keys = results["problem_keys"]
    problem_items = results["problem_items"]
    problem = {problem_keys[i]: problem_items[i] for i in range(len(problem_keys))}
    n_reps = results["n_reps"]
    n_samples = int(len(P_alive) / (2*problem["num_vars"]+2))

    for key, val in results.items():
        print(key, val)
        print("")

    print(problem)

    # obtain sobol indices for each statistic
    Si_plants_alive = sobol.analyze(problem, P_alive, print_to_console=True)
    Si_polls_alive = sobol.analyze(problem, A_alive, print_to_console=True)
    Si_all_alive = sobol.analyze(problem, P_alive+A_alive, print_to_console=True)
    Si_plants_abundance = sobol.analyze(problem, P_abundance, print_to_console=True)
    Si_polls_abundance = sobol.analyze(problem, A_abundance, print_to_console=True)

    axs = Si_plants_alive.plot()
    fig = plt.gcf()
    fig.set_size_inches(12, 16)
    axs[0].set_title("Total index")
    axs[1].set_title("First order index")
    axs[2].set_title("Second order index")
    axs[0].set_ylabel("Sensitivity index")
    axs[1].set_ylabel("Sensitivity index")
    axs[2].set_ylabel("Sensitivity index")
    axs[0].hlines(0, -1, 5, colors="black", linestyles="dashed")
    axs[1].hlines(0, -1, 5, colors="black", linestyles="dashed")
    axs[2].hlines(0, -1, 10, colors="black", linestyles="dashed")

    axs[0].set_xticks([0, 1, 2, 3, 4], [r"$q$", r"$N$", r"$D$", r"$\nu$", r"$\mu$"])
    axs[1].set_xticks([0, 1, 2, 3, 4], [r"$q$", r"$N$", r"$D$", r"$\nu$", r"$\mu$"])
    axs[2].set_xticks(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [r"$(q, N)$", r"$(q, D)$", r"$(q, \nu)$", r"$(q, \mu)$", r"$(N, D)$",
        r"$(N, \nu)$", r"$(N, \mu)$", r"$(D, \nu)$", r"$(D, \mu)$", r"$(\nu, \mu)$"]
    )
    for i in range(3):
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha='center')
        axs[i].get_legend().remove()
    pos = axs[0].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.1, 0.45, 0.55, 0.95
    axs[0].set_position(pos)
    pos = axs[1].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.55, 0.95, 0.55, 0.95
    axs[1].set_position(pos)
    pos = axs[2].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.1, 0.95, 0.1, 0.45
    axs[2].set_position(pos)
    plt.savefig(f"figures/sa_AM_fig0_{fnumber}.pdf")

    axs = Si_polls_alive.plot()
    fig = plt.gcf()
    fig.set_size_inches(12, 16)
    axs[0].set_title("Total index")
    axs[1].set_title("First order index")
    axs[2].set_title("Second order index")
    axs[0].set_ylabel("Sensitivity index")
    axs[0].hlines(0, -1, 5, colors="black", linestyles="dashed")
    axs[1].hlines(0, -1, 5, colors="black", linestyles="dashed")
    axs[2].hlines(0, -1, 10, colors="black", linestyles="dashed")
    axs[0].set_xticks([0, 1, 2, 3, 4], [r"$q$", r"$N$", r"$D$", r"$\nu$", r"$\mu$"])
    axs[1].set_xticks([0, 1, 2, 3, 4], [r"$q$", r"$N$", r"$D$", r"$\nu$", r"$\mu$"])
    axs[2].set_xticks(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [r"$(q, N)$", r"$(q, D)$", r"$(q, \nu)$", r"$(q, \mu)$", r"$(N, D)$",
        r"$(N, \nu)$", r"$(N, \mu)$", r"$(D, \nu)$", r"$(D, \mu)$", r"$(\nu, \mu)$"]
    )
    for i in range(3):
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha='center')
        axs[i].get_legend().remove()
    pos = axs[0].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.1, 0.45, 0.55, 0.95
    axs[0].set_position(pos)
    pos = axs[1].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.55, 0.95, 0.55, 0.95
    axs[1].set_position(pos)
    pos = axs[2].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.1, 0.95, 0.1, 0.45
    axs[2].set_position(pos)
    plt.savefig(f"figures/sa_AM_fig1_{fnumber}.pdf")

    axs = Si_plants_abundance.plot()
    fig = plt.gcf()
    fig.set_size_inches(12, 16)
    axs[0].set_title("Total index")
    axs[1].set_title("First order index")
    axs[2].set_title("Second order index")
    axs[0].set_ylabel("Sensitivity index")
    axs[0].hlines(0, -1, 5, colors="black", linestyles="dashed")
    axs[1].hlines(0, -1, 5, colors="black", linestyles="dashed")
    axs[2].hlines(0, -1, 10, colors="black", linestyles="dashed")
    axs[0].set_xticks([0, 1, 2, 3, 4], [r"$q$", r"$N$", r"$D$", r"$\nu$", r"$\mu$"])
    axs[1].set_xticks([0, 1, 2, 3, 4], [r"$q$", r"$N$", r"$D$", r"$\nu$", r"$\mu$"])
    axs[2].set_xticks(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [r"$(q, N)$", r"$(q, D)$", r"$(q, \nu)$", r"$(q, \mu)$", r"$(N, D)$",
        r"$(N, \nu)$", r"$(N, \mu)$", r"$(D, \nu)$", r"$(D, \mu)$", r"$(\nu, \mu)$"]
    )
    for i in range(3):
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha='center')
        axs[i].get_legend().remove()
    pos = axs[0].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.1, 0.45, 0.55, 0.95
    axs[0].set_position(pos)
    pos = axs[1].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.55, 0.95, 0.55, 0.95
    axs[1].set_position(pos)
    pos = axs[2].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.1, 0.95, 0.1, 0.45
    axs[2].set_position(pos)
    plt.savefig(f"figures/sa_AM_fig2_{fnumber}.pdf")

    axs = Si_polls_abundance.plot()
    fig = plt.gcf()
    fig.set_size_inches(12, 16)
    axs[0].set_title("Total index")
    axs[1].set_title("First order index")
    axs[2].set_title("Second order index")
    axs[0].set_ylabel("Sensitivity index")
    axs[0].hlines(0, -1, 5, colors="black", linestyles="dashed")
    axs[1].hlines(0, -1, 5, colors="black", linestyles="dashed")
    axs[2].hlines(0, -1, 10, colors="black", linestyles="dashed")
    axs[0].set_xticks([0, 1, 2, 3, 4], [r"$q$", r"$N$", r"$D$", r"$\nu$", r"$\mu$"])
    axs[1].set_xticks([0, 1, 2, 3, 4], [r"$q$", r"$N$", r"$D$", r"$\nu$", r"$\mu$"])
    axs[2].set_xticks(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [r"$(q, N)$", r"$(q, D)$", r"$(q, \nu)$", r"$(q, \mu)$", r"$(N, D)$",
        r"$(N, \nu)$", r"$(N, \mu)$", r"$(D, \nu)$", r"$(D, \mu)$", r"$(\nu, \mu)$"]
    )
    for i in range(3):
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha='center')
        axs[i].get_legend().remove()
    pos = axs[0].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.1, 0.45, 0.55, 0.95
    axs[0].set_position(pos)
    pos = axs[1].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.55, 0.95, 0.55, 0.95
    axs[1].set_position(pos)
    pos = axs[2].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.1, 0.95, 0.1, 0.45
    axs[2].set_position(pos)
    plt.savefig(f"figures/sa_AM_fig3_{fnumber}.pdf")

    axs = Si_all_alive.plot()
    fig = plt.gcf()
    fig.set_size_inches(12, 16)
    axs[0].set_title("Total index")
    axs[1].set_title("First order index")
    axs[2].set_title("Second order index")
    axs[0].set_ylabel("Sensitivity index")
    axs[1].set_ylabel("Sensitivity index")
    axs[2].set_ylabel("Sensitivity index")
    axs[0].hlines(0, -1, 5, colors="black", linestyles="dashed")
    axs[1].hlines(0, -1, 5, colors="black", linestyles="dashed")
    axs[2].hlines(0, -1, 10, colors="black", linestyles="dashed")
    axs[0].set_xticks([0, 1, 2, 3, 4], [r"$q$", r"$N$", r"$D$", r"$\nu$", r"$\mu$"])
    axs[1].set_xticks([0, 1, 2, 3, 4], [r"$q$", r"$N$", r"$D$", r"$\nu$", r"$\mu$"])
    axs[2].set_xticks(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [r"$(q, N)$", r"$(q, D)$", r"$(q, \nu)$", r"$(q, \mu)$", r"$(N, D)$",
        r"$(N, \nu)$", r"$(N, \mu)$", r"$(D, \nu)$", r"$(D, \mu)$", r"$(\nu, \mu)$"]
    )
    for i in range(3):
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha='center')
        axs[i].get_legend().remove()
    pos = axs[0].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.1, 0.45, 0.55, 0.95
    axs[0].set_position(pos)
    pos = axs[1].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.55, 0.95, 0.55, 0.95
    axs[1].set_position(pos)
    pos = axs[2].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.1, 0.95, 0.1, 0.45
    axs[2].set_position(pos)
    plt.savefig(f"figures/sa_AM_fig4_{fnumber}.pdf")


def run_sa_dA_Sinit(N, fnumber=0, parallel=False):
    """Perform the sensitivity analysis on the point of collapse
    N_REPS is number of times a single sampled parameter set is simulated.

    Parameters
    ----------
    N : int
        The number of samples per parameter. Total samples is N(2D + 2) with D the number
        of parameters varied.
    fnumber : int, optional
        The filenumber in which the results are saved
    parallelel : bool, optional
        If True, the sensitivity analysis is done parallel on all the cores available
    """
    # define problem
    problem = {
        "num_vars": 6,
        "names": ["q", "nestedness", "connectance", "nu", "S_init", "rate"],
        "bounds": [
            [0, 1],
            [0.2, 0.7],
            [0.15, 0.35],
            [0, 1],
            [0, 4],
            [0.0001, 1]
        ]
    }

    # generate parameter samples
    # in total N(2D + 2) samples (D is the amount of params, N samples per param)
    param_values = saltelli.sample(problem, N)

    n_params = param_values.shape[0]
    print(f"\nCalculating a total of {n_params} samples...")
    print("Expected runtime: ")
    print(f"\t{n_params*N_REPS*1.5:.2f} seconds")
    print(f"\t{n_params*N_REPS*1.5/(60):.2f} minutes")
    print(f"\t{n_params*N_REPS*1.5/(60*60):.2f} hours")

    # Parallel is faster as long as model simulation time is on average around
    # 0.1 seconds or more
    if parallel:
        # run model for each parameter set in parallel
        # results = np.zeros(param_values.shape[0])
        with mp.Pool(os.cpu_count()) as pool:
            results = np.asarray(pool.map(evaluate_model_dA_Sinit, param_values))

    else:
        # run model for each parameter set
        results = np.zeros(n_params)
        for i in range(param_values.shape[0]):
            print(f"{i+1} out of {param_values.shape[0]}")
            params = param_values[i]
            results[i] = evaluate_model_dA_Sinit(params)

    dA_crit = results

    # save results for later analysis
    fname = f"output/sa/sa_dA_Sinit_AM_{fnumber}.npz"
    np.savez(
        fname, dA_crit=dA_crit, problem_keys=np.asarray(list(problem.keys())),
        problem_items=np.asarray(list(problem.values()), dtype=object),
        n_reps=np.asarray(N_REPS)
    )


def evaluate_model_dA_Sinit(params):
    """Evaluates model for given parameters to calculate point of collapse

    Parameters
    ----------
    params : array
        Contains the values of q, nestedness, connectance, nu, S_init, rate
        to evaluate the model.
    Returns
    ---------
    dA_crit : float:
        Point of collapse averaged over N_REPS repetitions.
    """
    q, nestedness, connectance, nu, S_init, rate = params

    # maximum simulation time, even if equilibrium is not reached
    t_end = int(1e6)
    n_steps = int(1e5)
    N_p = 15
    N_a = 35

    def dA_rate(t, rate):
        return rate * t

    dA = {
        "func": dA_rate,
        "args": (rate, )
    }

    dA_crit_temp = np.zeros(N_REPS)
    for rep in range(N_REPS):

        AM = pc.AdaptiveModel(
            N_p, N_a, mu=0.0001, connectance=connectance, forbidden=0.3,
            nestedness=nestedness, network_type="nested", rng=None, seed=None, nu=nu,
            G=1, q=q, feasible=False, beta_trade_off=0.5
        )

        # set initial conditions
        y0 = np.full(AM.N, S_init, dtype=float)
        y0 = np.concatenate((y0, copy.deepcopy(AM.alpha.flatten())))

        # solve model until full collapse of pollinators, do this for a maximum time
        # (when da reaches 5 for lowest rate (0.0001)). Should be enough?
        sol = AM.solve(
            t_end, dA=dA, n_steps=n_steps, stop_on_collapse=True, save_period=None
        )
        collapse_time = sol.collapse_time

        if collapse_time is not None:
            dA_crit_temp[rep] = dA_rate(collapse_time, rate)

    dA_crit = np.mean(dA_crit_temp)

    return dA_crit


def analyze_sa_dA_Sinit(fnumber):
    """Calculate the sobol indices for the sensitivity analysis on the point of collapse
     and plot them.

    Parameters
    ----------
    fnumber : int
        The filenumber in which the results are saved
    """
    # load results of sa
    fname = f"output/sa/sa_dA_Sinit_AM_{fnumber}.npz"
    results = np.load(fname, allow_pickle=True)
    dA_crit = results["dA_crit"]
    problem_keys = results["problem_keys"]
    problem_items = results["problem_items"]
    problem = {problem_keys[i]: problem_items[i] for i in range(len(problem_keys))}
    # seed = results["seed"]

    for key, val in results.items():
        print(key, val)
        print("")

    print(problem)

    # obtain sobol indices for each statistic
    Si_dA = sobol.analyze(problem, dA_crit, print_to_console=True)

    axs = Si_dA.plot()
    fig = plt.gcf()
    fig.set_size_inches(12, 16)
    axs[0].set_title("Total index")
    axs[1].set_title("First order index")
    axs[2].set_title("Second order index")
    axs[0].set_ylabel("Sensitivity index")
    axs[1].set_ylabel("Sensitivity index")
    axs[2].set_ylabel("Sensitivity index")
    axs[0].hlines(0, -1, 6, colors="black", linestyles="dashed")
    axs[1].hlines(0, -1, 6, colors="black", linestyles="dashed")
    axs[2].hlines(0, -1, 15, colors="black", linestyles="dashed")
    axs[0].set_xticks(
        [0, 1, 2, 3, 4, 5],
        [r"$q$", r"$N$", r"$D$", r"$\nu$", r"$S^{\mathrm{\ init}}$", r"$\lambda$"]
    )
    axs[1].set_xticks(
        [0, 1, 2, 3, 4, 5],
        [r"$q$", r"$N$", r"$D$", r"$\nu$", r"$S^{\mathrm{\ init}}$", r"$\lambda$"]
    )
    axs[2].set_xticks(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [r"$(q, N)$", r"$(q, D)$", r"$(q, \nu)$", r"$(q, S^{\mathrm{\ init}})$",
        r"$(q, \lambda)$", r"$(N, D)$", r"$(N, \nu)$",
        r"$(N, S^{\mathrm{\ init}})$", r"$(N, \lambda)$", r"$(D, \nu)$",
        r"$(D, S^{\mathrm{\ init}})$", r"$(D, \lambda)$",
        r"$(\nu, S^{\mathrm{\ init}})$", r"$(\nu, \lambda)$",
        r"$(S^{\mathrm{\ init}}, \lambda)$"]
    )
    for i in range(3):
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha='center')
        axs[i].get_legend().remove()
    pos = axs[0].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.1, 0.45, 0.55, 0.95
    axs[0].set_position(pos)
    pos = axs[1].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.55, 0.95, 0.55, 0.95
    axs[1].set_position(pos)
    pos = axs[2].get_position()
    pos.x0, pos.x1, pos.y0, pos.y1 = 0.1, 0.95, 0.1, 0.45
    axs[2].set_position(pos)
    plt.savefig(f"figures/sa_dA_Sinit_AM_fig0_{fnumber}.pdf")

    return


def test_sa():
    """Function to test sensitivity analysis using Ishigami function """

    # define problem
    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi]
        ]
    }

    # generate parameter samples
    # in total N(2D + 2) samples (D is the amount of params, N samples per param)
    param_values = saltelli.sample(problem, 1024)
    print(param_values.shape[0])

    parallel = False

    def Ishigami(x, a=7, b=0.1):
        x1, x2, x3 = x
        return np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)

    # Works now. Is faster as long as model simulation time is on average around
    # 0.1 seconds or more
    if parallel:
        # run model for each parameter set in parallel
        # results = np.zeros(param_values.shape[0])
        with mp.Pool(os.cpu_count()) as pool:
            results = np.asarray(pool.map(Ishigami, param_values))

    else:
        # run model for each parameter set
        results = np.zeros(param_values.shape[0])
        for i in tqdm(range(param_values.shape[0])):
            params = param_values[i]
            results[i] = Ishigami(params)

    # obtain sobol indices
    Si = sobol.analyze(problem, results, print_to_console=True)

    Si.plot()


if __name__ == "__main__":
    main()
