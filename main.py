#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 19/01/2022
# ---------------------------------------------------------------------------
""" main.py

Model of pollinator communities (pollcomms)
"""
# ---------------------------------------------------------------------------
import copy
import os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import experiments as exp

import pollcomm as pc
import visualization as vis

# check if ./output and ./figures directories exist. If not, create them
os.makedirs(os.path.join(os.getcwd(), "output"), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "figures"), exist_ok=True)


def main():

    t0 = timer()
    # general simulation parameters
    plt.rcParams.update({"font.size": 14})  # to make figures more readable

    seed = np.random.SeedSequence().generate_state(1)[0]
    seed = 3892699245
    rng = np.random.default_rng(seed)

    N_p = 25
    N_a = 25
    mu = 0.0001
    connectance = 0.15
    forbidden = 0.3
    nestedness = 0.6
    network_type = "nested"
    t_end = 500
    dA = 0

    # exp.time_sol_AM_BM_VM(dA=dA, seed=seed, save_fig=False)

    # exp.AM_beta_alpha(dA=dA, seed=seed, save_fig=False)
    # exp.VM_alpha(dA=dA, seed=seed, save_fig=True)

    # exp.AM_q(seed=seed, save_fig=True, recalculate=True)

    # exp.state_space_AM_BM(seed=seed, recalculate=False)
    # exp.state_space_AM_BM_VM(seed=seed, recalculate=False)

    # exp.state_space_rate_AM_BM_VM(seed=seed, recalculate=False)

    # exp.state_space_BM(plot=True, save_fig=False)
    # exp.state_space_C_BM(plot=True, save_fig=True)
    # exp.state_space_rate_BM(plot=True, save_fig=False)

    # exp.state_space_AM(G=1, q=0.1, plot=True, save_fig=False)
    # exp.state_space_rate_AM(G=1, plot=True, save_fig=True)

    # exp.state_space_ARM(G=1, seed=seed, plot=True)

    # exp.state_space_VM(plot=True, save_fig=True)
    # exp.state_space_rate_VM(plot=True, save_fig=True)

    # exp.state_space_abundance_rate_AM(
    #     seed=seed, plot=True, save_fig=True, recalculate=True, q=0.1
    # )

    # exp.state_space_abundance_env_AM(
    #     seed=seed, plot=True, save_fig=True, recalculate=True, q=0, G=0, nu=0.5
    # )
    # exp.state_space_abundance_env_AM(
    #     seed=seed, plot=True, save_fig=True, recalculate=True, q=0.1
    # )
    #
    # exp.state_space_abundance_env_AM(
    #     seed=seed, plot=True, save_fig=True, recalculate=True, q=0.1
    # )
    # exp.state_space_abundance_env_AM(
    #     seed=seed, plot=True, save_fig=True, recalculate=True, q=1
    # )

    # exp.state_space_abundance_env_rate_AM(
    #     seed=seed, plot=True, save_fig=True, recalculate=True, q=0.1
    # )
    # exp.state_space_abundance_env_rate_AM(
    #     seed=seed, plot=True, save_fig=True, recalculate=True, q=0.2
    # )

    # exp.state_space_abundance_env_rate_AM(
    #     seed=seed, plot=True, save_fig=True, recalculate=True, q=0.5
    # )
    # exp.state_space_abundance_env_rate_AM(
    #     seed=seed, plot=True, save_fig=True, recalculate=True, q=1
    # )

    # print(f"Total time: {timer()-t0:.2f} seconds")
    # return plt.show()

    # BM = pc.BaseModel(
    #     N_p, N_a, mu, connectance, forbidden, nestedness, rng=rng, seed=seed,
    #     feasible=True
    # )
    # sol = BM.solve(t_end, dA=dA)
    # vis.plot_time_sol_mean(BM, dA=dA)
    # vis.plot_time_sol_number_alive(BM, dA=dA)
    # vis.plot_time_sol_pollcomm(sol, BM, dA=dA)

    # AM = pc.AdaptiveModel(
    #     N_p, N_a, mu, connectance, forbidden, nestedness, network_type="001", rng=rng,
    #     seed=seed, q=0, G
    # )
    connectance = 0.15
    nestedness = 0.6
    AM = pc.AdaptiveModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type="nested", rng=rng,
        seed=seed, nu=1, G=1, q=0, feasible=True, feasible_iters=50
    )
    result = exp.equilibrium("", AM)
    # print(result.shape)
    # print(result[:AM.N])

    sol = AM.solve(t_end, stop_on_collapse=True, dA=0, n_steps=1000, y0=AM.y_all_end)
    vis.plot_time_sol_pollcomm(sol, AM, save_fig=True)
    return plt.show()

    t_end, n_steps = 250, 1000
    def dA_rate(t, rate):
        return rate * t
    dA = {
        "func": dA_rate,
        "args": (0.3, )
    }
    y0 = np.full(AM.N, 0.25, dtype=float)
    y0 = np.concatenate((y0, AM.alpha.flatten()))
    sol = AM.solve(t_end, stop_on_collapse=True, dA=dA, n_steps=n_steps, y0=y0)
    vis.plot_time_sol_mean(AM)
    vis.plot_time_sol_number_alive(AM)
    alpha_init = AM.alpha
    alpha_end = sol.y_partial[:, -1].reshape((AM.N_p, AM.N_a))
    beta_P, beta_A = AM.beta_P, AM.beta_A

    # print(
    #     np.nan_to_num(alpha_init/beta_P)[np.nan_to_num(alpha_init/beta_P).nonzero()] -
    #     np.nan_to_num(alpha_init/beta_A)[np.nan_to_num(alpha_init/beta_A).nonzero()]
    # )
    #
    # print(np.nan_to_num(alpha_init/beta_P)[np.nan_to_num(alpha_init/beta_P).nonzero()].mean())
    # print(np.nan_to_num(alpha_end/beta_P)[np.nan_to_num(alpha_end/beta_P).nonzero()].mean())
    #
    # print(np.nan_to_num(alpha_init/beta_A)[np.nan_to_num(alpha_init/beta_A).nonzero()].mean())
    # print(np.nan_to_num(alpha_end/beta_A)[np.nan_to_num(alpha_end/beta_A).nonzero()].mean())

    # plt.matshow(3*beta_P*alpha_init, origin="lower")
    # plt.colorbar()
    # plt.matshow(beta_A*alpha_init, origin="lower")
    # plt.colorbar()
    # # return plt.show()
    vis.plot_AM_alpha_init_alpha_end_beta(alpha_init, alpha_end, beta_P, beta_A, save_fig=True)
    vis.plot_time_sol_pollcomm(sol, AM, save_fig=True)
    vis.plot_degree_abundance(AM, save_fig=True)

    # ARM = pc.AdaptiveResourceModel(
    #     N_p, N_a, mu, connectance, forbidden, nestedness, network_type="nested", rng=rng,
    #     seed=seed, nu=0.001, G=0.1, q=0, feasible=True
    # )
    # sol = ARM.solve(t_end, dA=dA)
    # vis.plot_time_sol_mean(ARM, dA=dA)
    # vis.plot_time_sol_number_alive(ARM, dA=dA)
    # alpha_init = ARM.alpha
    # alpha_end = sol.y_partial[:, -1].reshape((ARM.N_p, ARM.N_a))
    # beta = ARM.beta
    # vis.plot_AM_alpha_init_alpha_end_beta(alpha_init, alpha_end, beta)
    # vis.plot_time_sol_pollcomm(sol, ARM, dA=dA)
    return plt.show()

    # AM = pc.AdaptiveModel(
    #     N_p, N_a, mu, connectance, forbidden, nestedness, rng=rng, seed=seed, G=1, q=1
    # )
    # sol = AM.solve(t_end, dA=0)
    # vis.plot_time_sol_AM(sol, AM, t_end=t_end, dA=dA)
    # return plt.show()

    # ARM = pc.AdaptiveResourceModel(
    #     N_p, N_a, mu, connectance, forbidden, nestedness, rng=rng, seed=seed, G=1,
    #     q=1, nu=0.05
    # )
    # sol = ARM.solve(t_end, dA=dA)
    # vis.plot_time_sol_ARM(sol, ARM, t_end=t_end, dA=dA)
    #
    # alpha_init = ARM.alpha
    # alpha_end = sol.y_partial[:, -1].reshape((ARM.N_p, ARM.N_a))
    # beta = ARM.beta
    # vis.plot_AM_alpha_init_alpha_end_beta(alpha_init, alpha_end, beta)
    # return plt.show()

    # vis.plot_time_sol_AM(sol, AM, t_end)
    # network = AM.network
    # alpha = sol.y_partial[:, -1].reshape((AM.N_p, AM.N_a))
    # inds1 = np.argsort(network.sum(axis=0))
    # plt.figure()
    # r_a = AM.r_a[inds1[::-1]]
    # plt.scatter(range(len(r_a)), r_a)
    #
    # inds2 = np.argsort(network.sum(axis=1))
    # plt.figure()
    # r_p = AM.r_p[inds2[::-1]]
    # plt.scatter(range(len(r_p)), r_p)
    #
    # beta = AM.beta
    # forbidden_network = AM.forbidden_network
    #
    # vis.plot_alpha_beta_network(network, alpha, beta, forbidden_network, save_fig=False)
    #
    #
    # return plt.show()

    VM = ValdovinosModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, seed=seed
    )

    y0 = np.full(VM.N_p, 1, dtype=float)
    y0 = np.concatenate((y0, np.full(VM.N_a, 3, dtype=float)))
    y0 = np.concatenate((y0, np.full(VM.N_p, 0.2, dtype=float)))
    alpha = copy.deepcopy(VM.alpha0)
    y0 = np.concatenate((y0, alpha.flatten()))
    sol = VM.solve(t_end, dA=dA, y0=y0, n_steps=int(1e5))
    vis.plot_time_sol_VM(sol, VM)

    VM = ValdovinosModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, rng=rng, seed=seed
    )

    y0 = np.full(VM.N_p, 0.1, dtype=float)
    y0 = np.concatenate((y0, np.full(VM.N_a, 0.1, dtype=float)))
    y0 = np.concatenate((y0, np.full(VM.N_p, 0.05, dtype=float)))
    alpha = copy.deepcopy(VM.alpha0)
    y0 = np.concatenate((y0, alpha.flatten()))
    sol = VM.solve(t_end, dA=dA, y0=y0, n_steps=int(1e5))
    vis.plot_time_sol_VM(sol, VM)

    #
    # network = VM.network
    # alpha = VM.alpha0
    # forbidden_network = VM.forbidden_network
    # vis.plot_alpha_beta_network(network, alpha, copy.deepcopy(alpha), forbidden_network, save_fig=False)

    return plt.show()

    def dA_rate(t, r, dA_max, t_init):
        t_max = dA_max / r + t_init
        if t < t_init:
            return 0
        elif t <= t_max:
            return r * (t - t_init)
        elif t > t_max:
            return dA_max
    rate = 0.00001
    t_init = 250
    dA_max = dA
    dA_dict = {
        "func": dA_rate,
        "args": (rate, dA_max, t_init)
    }
    t_end = dA_max / rate + t_init + 350
    print(t_end)

    sol = VM.solve(t_end, dA=dA, n_steps=int(1e4))
    vis.plot_time_sol_VM(sol, VM)
    sol = VM.solve(t_end, dA=dA_dict, n_steps=int(1e4))
    vis.plot_time_sol_VM(sol, VM, dA_dict)
    return plt.show()

    # ARM = AdaptiveExplicitResourceModel(
    #     N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng, seed,
    #     nu = 0.05, G=1
    # )
    #
    # def dA_rate(t, r, dA_max, t_init):
    #     t_max = dA_max / r + t_init
    #     if t < t_init:
    #         return 0
    #     elif t <= t_max:
    #         return r * (t - t_init)
    #     elif t > t_max:
    #         return dA_max
    #
    # rate = 0.01
    # t_init = 100
    # dA_max = 1.2
    # dA_dict = {
    #     "func": dA_rate,
    #     "args": (rate, dA_max, t_init)
    # }
    # t_end = dA_max / rate + t_init + 50
    #
    # # ARM_sol = ARM.solve(t_end, dA=0)
    # # y0 = np.concatenate((ARM_sol.y[:ARM.N, -1], ARM_sol.y_partial[:, -1], ARM_sol.y[ARM.N:, -1]))
    # ARM_sol = ARM.solve(t_end, dA=dA_dict, y0=None)
    #
    # vis.plot_time_sol_ARM(ARM_sol, ARM, t_end=t_end, t_remove=0.3, dA=dA_dict)
    #
    # network = ARM.network
    # alpha = ARM_sol.y_partial[:, -1].reshape((ARM.N_p, ARM.N_a))
    # alpha[alpha < 0.01] = 0
    # alpha[alpha >= 1/N_p]
    # beta = ARM.beta
    # forbidden_network = ARM.forbidden_network
    # vis.plot_alpha_beta_network(network, alpha, beta, forbidden_network, save_fig=True)
    #
    # return plt.show()

    ARM = AdaptiveModel(
        N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng, seed,
        G=1, nu=0.3
    )

    # print((ARM.alpha * ARM.beta).sum(axis=0))
    # print((ARM.alpha * ARM.beta).sum(axis=1))
    # return
    # network = ARM.network
    # alpha = ARM.beta
    # beta = ARM.beta
    # forbidden_network = ARM.forbidden_network
    # vis.plot_alpha_beta_network(network, alpha, beta, forbidden_network, save_fig=True)

    def dA_rate(t, r, dA_max, t_init):
        t_max = dA_max / r + t_init
        if t < t_init:
            return 0
        elif t <= t_max:
            return r * (t - t_init)
        elif t > t_max:
            return dA_max

    rate = 0.01
    t_init = 100
    dA_max = 0.8
    dA_dict = {
        "func": dA_rate,
        "args": (rate, dA_max, t_init)
    }
    t_end = dA_max / rate + t_init + 50
    # dA_dict = 0
    # ARM_sol = ARM.solve(t_end, dA=0)
    # y0 = np.concatenate((ARM_sol.y[:ARM.N, -1], ARM_sol.y_partial[:, -1], ARM_sol.y[ARM.N:, -1]))
    ARM_sol = ARM.solve(t_end, dA=dA_dict, y0=None)

    vis.plot_time_sol_AM(ARM_sol, ARM, t_end=t_end, t_remove=0.3, dA=None)


    rate = 0.1
    t_init = 100
    dA_max = 0.3
    dA_dict = {
        "func": dA_rate,
        "args": (rate, dA_max, t_init)
    }
    t_end = dA_max / rate + t_init + 50
    # dA_dict = 0
    # ARM_sol = ARM.solve(t_end, dA=0)
    # y0 = np.concatenate((ARM_sol.y[:ARM.N, -1], ARM_sol.y_partial[:, -1], ARM_sol.y[ARM.N:, -1]))
    ARM_sol = ARM.solve(t_end, dA=dA_dict, y0=None)

    vis.plot_time_sol_AM(ARM_sol, ARM, t_end=t_end, t_remove=0.3, dA=None)


    network = ARM.network
    alpha = ARM_sol.y_partial[:, -1].reshape((ARM.N_p, ARM.N_a))
    beta = ARM.beta
    forbidden_network = ARM.forbidden_network
    vis.plot_alpha_beta_network(network, alpha, beta, forbidden_network, save_fig=True)
    # alpha[alpha < 1/N_p] = 0
    # alpha[alpha >= 1/N_p] = 1
    # vis.plot_alpha_beta_network(network, alpha, beta, forbidden_network, save_fig=True)


    return plt.show()

    # exp.state_space_rate_AM(seed, plot=True, fname="output/AM_state_rates1")
    # exp.state_space_rate_BM(seed, plot=True, fname="output/BM_state_rates1")
    # vis.plot_state_space_rate_AM_BM(
    #     "output/AM_state_rates1.npz", "output/BM_state_rates1.npz"
    # )

    # exp.state_space_BM(
    #     seed, plot=False, save_fig=False, fname="output/AM1.npz"
    # )
    # exp.state_space_BM(
    #     seed, plot=False, save_fig=False, fname="output/BM1.npz"
    # )
    # vis.plot_state_space_AM_BM("output/AM1.npz", "output/BM1.npz")

    # create two identical rng's for fair comparison between models
    # seed_seq = np.random.SeedSequence()
    # seed_seq = 345679876567 # fast seed to generate networks with
    # rng1 = np.random.default_rng(seed_seq)
    # rng2 = np.random.default_rng(seed_seq)
    #
    # def func(t, r):
    #     t0 = 20
    #     dA_max = 3
    #     if t <= t0:
    #         return 0
    #     elif t > t0:
    #         if r * (t-t0) < dA_max:
    #             return r * (t-t0)
    #         else:
    #             return dA_max
    # rs = [0, 0.05, 0.2, 0.5]
    # dA = {
    #     "func": func,
    #     "args": None
    # }
    # dAs = []
    # for r in rs:
    #     dA_copy = copy.copy(dA)
    #     dA_copy["args"] = r
    #     dAs.append(dA_copy)
    #
    #
    # exp.AM_four_dAs(dAs=dAs, G=1, rng=rng1, plot_network=False, t_end=75)
    # dAs = [2, 2.8, 3, 3.5]
    # exp.AM_four_dAs(dAs=dAs, G=1, rng=rng2, plot_network=False, t_end=75)

    # exp.plot_alpha_beta_network(save_fig=True, rng=None)

    # N_p = 20
    # N_a = 20
    # mu = 0
    # connectance = 0.15
    # forbidden = 0.3
    # nestedness = 0.6
    # network_type = "nested"
    # t_end = 50
    # dA = 0
    #
    # seed = None
    # rng = np.random.default_rng(seed)
    # BM = BaseModel(
    #     N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng, seed
    # )
    # AM = AdaptiveModel(
    #     N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng, seed
    # )
    #
    # BM_sol = BM.solve(t_end)
    # AM_sol = AM.solve(t_end)
    #
    # vis.plot_time_sol_pollcomm(sol, pollcomm, dA=dA)
    #
    # sol = pollcomm.solve(t_end, dA=dA, n_steps=int(1e5), save_period=200)
    # vis.plot_time_sol_pollcomm(sol, pollcomm, dA=dA)

    # vis.anim_alpha(sol, pollcomm, dA=dA)

    # if rng is None:
    #     rng = np.random.default_rng()
    #
    # N_p = 20
    # N_a = 20
    # mu = 0
    # connectance = 0.15
    # forbidden = 0.3
    # nestedness = 0.6
    # network_type = "nested"
    # t_end = 50
    # G = 1
    #
    # AM = AdaptiveModel(
    #     N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng=rng, G=G
    # )
    #
    # alpha = AM.alpha
    # beta = AM.beta
    # network = AM.network
    # forbidden_network = AM.forbidden_network
    #
    # exp.plot_alpha_beta_network(network, alpha, beta, forbidden_network)

    return plt.show()


if __name__ == "__main__":
    main()
