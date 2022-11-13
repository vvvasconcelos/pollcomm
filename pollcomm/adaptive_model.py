#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 17/04/2022
# ---------------------------------------------------------------------------
""" adaptive_model.py

Implementation of Adaptive model

"""
# ---------------------------------------------------------------------------
import copy
from timeit import default_timer as timer

import numba
import numpy as np

from pollcomm.ode_solver import solve_ode
from pollcomm.pollcomm_class import PollcommMutualismCompetition

__all__ = ["AdaptiveModel"]


class AdaptiveModel(PollcommMutualismCompetition):
    def __init__(
        self, N_p=25, N_a=25, mu=0.0001, connectance=0.15, forbidden=0.3, nestedness=0.6,
        network_type="nested", rng=None, seed=None, nu=0.1, G=1, q=1, feasible=True,
        beta_trade_off=0.5, feasible_iters=100
    ):
        # t0 = timer()
        super().__init__(
            N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng, seed
        )

        self.beta_trade_off = beta_trade_off    # normalization factor beta matrices
        self.G = G      # AF speed factor
        self.nu = nu    # AF normalizing factor
        self.q = q      # excludability parameter: 1 for supply-demand, 0 for phi = P

        self.feasible = feasible    # whether params should give feasible solution
        self.feasible_iters = feasible_iters    # amount of iterations for feasible sol

        # whether network with sampled parameters is feasible
        self.is_feasible = None

        # set the parameters (alpha, beta, C, r, and h) for the model
        self.set_params()

        # tf = timer()
        # print(f"Created AdaptiveModel instance. Time elapsed: {tf-t0:.5f} seconds\n")

    def __repr__(self):
        return "AM"

    def equilibrium(self, dA=0, y0=None):

        t_end = int(1e6)
        n_steps = int(1e5)
        self.solve(
            t_end, n_steps=n_steps, stop_on_equilibrium=True, save_period=0, dA=dA, y0=y0
        )

        return self.y_all_end

    def find_dA_collapse(self, dA_step=0.02, y0=None):

        print("\nCalculating dA collapse...")
        if y0 is None:
            y0 = self.equilibrium()
        dA_found = False
        dA = 0
        while not dA_found:

            print(f"dA = {dA:.2f}")
            self.solve(
                1000, n_steps=1000, dA=dA, y0=y0, save_period=0, stop_on_equilibrium=True
            )
            if (self.y[self.N_p:self.N, -1] < 0.01).all():
                return dA

            # new starting conditions
            y0 = self.y[:, -1]
            y0 = np.concatenate((y0, self.y_partial[:, -1]))

            # increase dA
            dA += dA_step

    def ode(self, t, z, dA):
        """Full set of ODEs for pollcomm with dA either fixed (int, float) or as a dict
        in which a function is defined (dA(t, *args))

        Returns:
            ODEs [np array]: full set of ODEs for pollcomm. Order: plants, pollinators
        """
        if isinstance(dA, (int, float)):
            _dA = dA
        elif isinstance(dA, (dict)):
            _dA = dA["func"](t, *dA.get("args", None))

        P = z[:self.N_p]
        A = z[self.N_p:self.N]
        alpha = z[self.N:].reshape((self.N_p, self.N_a))

        # precalculating these term, since they are used more than once
        alpha_beta_A_prod = self._alpha_beta_A_prod(alpha, self.beta_A, A)
        alpha_beta_phi_prod = self._alpha_beta_phi_prod(
            alpha, self.beta_A, P, alpha_beta_A_prod
        )

        plant_ODEs = P * (
            self.r_p +
            self._mutualism(self._alpha_beta_A_prod(alpha, self.beta_P, A), self.h_p) -
            self._competition_vectorized(P, self.C_p)
        )  + self.mu

        poll_ODEs = A * (
            self.r_a - _dA + self._mutualism(alpha_beta_phi_prod, self.h_a) -
            self._competition_vectorized(A, self.C_a)
        ) + self.mu

        if self.nu == 1 or self.G == 0:
            alpha_ODEs = np.zeros((self.N_p, self.N_a))
        else:
            alpha_ODEs = self.G * ((1 - self.nu) * alpha *
                (
                    self.beta_A *
                    self._phi(P, alpha_beta_A_prod)[:, np.newaxis] -
                    alpha_beta_phi_prod
                ) + _nu_term_numba(alpha, self.beta_A, self.N_p, self.N_a, self.nu)
            )

        return np.concatenate((plant_ODEs, poll_ODEs, alpha_ODEs.flatten()))

    def set_params(self):
        """Set the parameters for the model. Can be used to set new parameters for
        same network by calling this method outside of the class
        (so that the current network is being re-used)
        """
        # always reset solution when sampling new parameters to avoid confusion
        self.reset_sol()
        self.is_feasible = None

        # sample parameters until we find a feasible solution for the given network
        # (meaning that all species are alive in equilibrium for dA=0)
        if self.feasible:
            for i in range(self.feasible_iters):
                self.reset_sol()
                self._sample_params()

                # we only need to get the solution for two time points (n_steps=2),
                # since we are only interested in model outcome at t_end
                sol = self.solve(
                    1000, n_steps=1000, stop_on_equilibrium=True, stop_on_collapse=True,
                    save_period=None
                )
                self.set_sol(sol)
                if self.is_all_alive()[-1]:
                    self.is_feasible = True
                    print("Found network with feasible parameters!")
                    break

            else:
                self.is_feasible = False
                print((
                    f"No feasible network found in {self.feasible_iters} iterations",
                    "Continue with the last sampled (not feasible) network"
                ))
        else:
            # sample parameters once and use these (don't check feasibility)
            self._sample_params()

    def solve(
        self, t_end, dA=0, n_steps=int(1e5), y0=None, save_period=0,
        stop_on_collapse=False, extinct_threshold=0.01, stop_on_equilibrium=False,
        equi_tol=1e-7
    ):
        """Numerical solver of pollcomm ODEs. Makes use of solve_ivp() function of scipy

        Returns:
            sol [obj]: numerical solution of ODE in a scipy Bunch object
        """
        t_span = (0, t_end)

        # If no initial conditions are provided, use default initial conditions.
        if y0 is None:
            y0 = np.full(self.N, 1, dtype=float)
            y0 = np.concatenate((y0, self.alpha.flatten()))

        # save alpha matrix only at the given save_period intervals,
        # otherwise it will take up a lot of memory once n_steps becomes large
        save_partial = {
            "ind": (self.N, y0.shape[0]-1),
            "save_period": save_period
        }

        t0 = timer()
        # print("Solving...")
        sol = solve_ode(
            self.ode, t_span, y0, n_steps, args=(dA,), save_partial=save_partial,
            rtol=1e-4, atol=1e-7, method="LSODA", stop_on_collapse=stop_on_collapse,
            N_p=self.N_p, N_a=self.N_a, extinct_threshold=extinct_threshold,
            stop_on_equilibrium=stop_on_equilibrium, equi_tol=equi_tol
        )
        # print(f"Solved Adaptive Model in {timer()-t0:.2f} seconds...\n")
        self.reset_sol()
        self.set_sol(sol)
        return sol

    def ode_dA(self, t, z, rate, dA_max):
        """Full set of ODEs for pollcomm with dA either fixed (int, float) or as a dict
        in which a function is defined (dA(t, *args))

        Returns:
            ODEs [np array]: full set of ODEs for pollcomm. Order: plants, pollinators
        """

        P = z[:self.N_p]
        A = z[self.N_p:self.N]
        alpha = z[self.N:self.N+self.N_p*self.N_a].reshape((self.N_p, self.N_a))
        dA = z[-1]

        if dA < dA_max:
            dA_ODE = np.array([rate])
        else:
            dA_ODE = np.array([0])

        # precalculating these term, since they are used more than once
        alpha_beta_A_prod = self._alpha_beta_A_prod(alpha, self.beta_A, A)
        alpha_beta_phi_prod = self._alpha_beta_phi_prod(
            alpha, self.beta_A, P, alpha_beta_A_prod
        )

        plant_ODEs = P * (
            self.r_p +
            self._mutualism(self._alpha_beta_A_prod(alpha, self.beta_P, A), self.h_p) -
            self._competition_vectorized(P, self.C_p)
        )  + self.mu

        poll_ODEs = A * (
            self.r_a - dA + self._mutualism(alpha_beta_phi_prod, self.h_a) -
            self._competition_vectorized(A, self.C_a)
        ) + self.mu

        if self.nu == 1 or self.G == 0:
            alpha_ODEs = np.zeros((self.N_p, self.N_a))
        else:
            alpha_ODEs = self.G * ((1 - self.nu) * alpha *
                (
                    self.beta_A *
                    self._phi(P, alpha_beta_A_prod)[:, np.newaxis] -
                    alpha_beta_phi_prod
                ) + _nu_term_numba(alpha, self.beta_A, self.N_p, self.N_a, self.nu)
            )

        return np.concatenate((plant_ODEs, poll_ODEs, alpha_ODEs.flatten(), dA_ODE))

    def solve_dA(
        self, t_end, rate=0, dA_max=0, n_steps=int(1e5), y0=None, save_period=0,
        stop_on_collapse=False, extinct_threshold=0.01, stop_on_equilibrium=False,
        equi_tol=1e-7
    ):
        """Numerical solver of pollcomm ODEs. Makes use of solve_ivp() function of scipy

        Returns:
            sol [obj]: numerical solution of ODE in a scipy Bunch object
        """
        t_span = (0, t_end)

        # If no initial conditions are provided, use default initial conditions.
        if y0 is None:
            y0 = np.full(self.N, 1, dtype=float)
            y0 = np.concatenate((y0, self.alpha.flatten()))
            y0 = np.concatenate((y0, np.array([0])))

        # save alpha matrix only at the given save_period intervals,
        # otherwise it will take up a lot of memory once n_steps becomes large
        save_partial = {
            "ind": (self.N, y0.shape[0]-2),
            "save_period": save_period
        }

        t0 = timer()
        # print("Solving...")
        sol = solve_ode(
            self.ode_dA, t_span, y0, n_steps, args=(rate, dA_max), save_partial=save_partial,
            rtol=1e-4, atol=1e-7, method="LSODA", stop_on_collapse=stop_on_collapse,
            N_p=self.N_p, N_a=self.N_a, extinct_threshold=extinct_threshold,
            stop_on_equilibrium=stop_on_equilibrium, equi_tol=equi_tol
        )
        # print(f"Solved Adaptive Model in {timer()-t0:.2f} seconds...\n")
        self.reset_sol()
        self.set_sol(sol)
        return sol

    def _alpha_beta_A_prod(self, alpha, beta, A):
        """Calculates np.sum(
            [alpha[i, j] * self.beta[i, j] * A[j] for j in range(self.N_a)]
        ) in a vectorized manner for all i in N_p
        """
        return (alpha * beta) @ A

    def _alpha_beta_phi_prod(self, alpha, beta, P, alpha_beta_A_prod):
        """Calculates np.sum(
            [pollcomm.alpha[i, j] * pollcomm.beta[i, j] *
            pollcomm._phi(P, alpha_beta_A_prod, i) for i in range(pollcomm.N_p)]
        ) in a vectorized manner for all j in N_a
        """
        phis = self._phi(P, alpha_beta_A_prod)
        return (alpha * beta).T @ phis

    def _generate_alpha(self):
        """Generate alpha (foraging effort) matrix according to network layout,
        normalized such that sum_i alpha_ij = 1

        Returns:
            alpha [np array]: alpha matrix
        """
        # alpha = np.zeros((self.N_p, self.N_a))
        # for i in range(self.N_p):
        #     for j in range(self.N_a):
        #         try:
        #             alpha[i, j] = self.network[i, j]/(
        #                 np.sum(self.network[:, j])**self.beta_trade_off
        #             )
        #         except ZeroDivisionError:
        #             alpha[i, j] = 0
        #
        # return alpha
        return self.network/self.network.sum(axis=0)

    def _generate_alpha_ones(self):
        """Generate alpha (foraging effort) matrix equal full with ones. Coupled with
        self._generate_beta(0.8, 1.2) this is equal to the base model

        Returns:
            alpha [np array]: alpha matrix
        """
        return np.ones((self.N_p, self.N_a))

    def _generate_beta_forbidden(self, low, high):
        """Generate beta (inherent trait matching) matrices with beta zero only
        where there are forbidden links present (i.e. self.forbidden_network == 1)

        Args:
            low [float]: lowest value from uniform sample interval
            high [float]: highest value from uniform sample interval
            Optional:
                Norm [bool]: whether to normalize over the plants
        Returns:
            beta_P [np array]: beta mutualism matrix for plants
            beta_A [np array]: beta mutualism matrix for pollinators
        """
        beta_P = np.zeros((self.N_p, self.N_a))
        beta_0 = self.rng.uniform(low, high, size=(self.N_p, self.N_a))
        for i in range(self.N_p):
            for j in range(self.N_a):
                    if self.forbidden_network[i, j] == 0:
                        beta_P[i, j] = beta_0[i, j] / np.sum(
                            self.network[i, :]
                        )**1

        beta_A = np.zeros((self.N_p, self.N_a))
        beta_0 = self.rng.uniform(low, high, size=(self.N_p, self.N_a))
        for i in range(self.N_p):
            for j in range(self.N_a):
                    if self.forbidden_network[i, j] == 0:
                        beta_A[i, j] = beta_0[i, j] / np.sum(
                            self.network[i, :]
                        )**1
        return beta_P, beta_A

        #
        # beta_A = np.zeros((self.N_p, self.N_a))
        # for i in range(self.N_p):
        #     for j in range(self.N_a):
        #         if self.forbidden_network[i, j] == 0:
        #             beta_A[i, j] = self.rng.uniform(low, high)
        #
        # beta_P = copy.deepcopy(beta_A)
        # if norm:
        #     beta_A /= beta_A.sum(axis=0)
        #
        # return beta_P, beta_A

    def _generate_beta_ones(self):
        """Generate beta (inherent trait matching) matrices filled with ones

        Returns:
            beta_P [np array]: beta mutualism matrix for plants
            beta_A [np array]: beta mutualism matrix for pollinators
        """
        return np.ones((self.N_p, self.N_a)), np.ones((self.N_p, self.N_a))

    def _generate_beta(self, low, high):
        """Generate beta (mutualism) matrices from uniform samples according to network
        topology (beta only nonzero where self.network == 1)

        The matriced are normalized by the number of connections to the power of
        self.beta_trade_off. For a properly normalized matrix, self.beta_trade_off
        should equal 1. For completely unnormalized, self.beta_trade_off should be 0.

        Args:
            low [float]: lowest value from uniform sample interval
            high [float]: highest value from uniform sample interval

        Returns:
            beta_P [np array]: beta mutualism matrix for plants
            beta_A [np array]: beta mutualism matrix for pollinators
        """
        beta_P = np.zeros((self.N_p, self.N_a))
        beta_0 = self.rng.uniform(low, high, size=(self.N_p, self.N_a))
        for i in range(self.N_p):
            for j in range(self.N_a):
                    if self.network[i, j] != 0:
                        beta_P[i, j] = beta_0[i, j] / np.sum(
                            self.network[i, :]
                        )**self.beta_trade_off

        beta_A = np.zeros((self.N_p, self.N_a))
        beta_0 = self.rng.uniform(low, high, size=(self.N_p, self.N_a))
        for i in range(self.N_p):
            for j in range(self.N_a):
                    if self.network[i, j] != 0:
                        # beta_A[i, j] = beta_0[i, j] / np.sum(
                        #     self.network[i, :]
                        # )**self.beta_trade_off
                        beta_A[i, j] = beta_0[i, j] / np.sum(
                            self.network[:, j]
                        )**self.beta_trade_off
        return beta_P, beta_A

    def _generate_beta_normed(self, low, high):
        """Generate beta (mutualism) matrices from uniform samples according to network
        topology (beta only nonzero where self.network == 1)

        The matriced are normalized by the number of connections to the power of
        self.beta_trade_off. For a properly normalized matrix, self.beta_trade_off
        should equal 1. For completely unnormalized, self.beta_trade_off should be 0.

        Args:
            low [float]: lowest value from uniform sample interval
            high [float]: highest value from uniform sample interval

        Returns:
            beta_P [np array]: beta mutualism matrix for plants
            beta_A [np array]: beta mutualism matrix for pollinators
        """
        beta_P = np.zeros((self.N_p, self.N_a))
        beta_0 = self.rng.uniform(low, high, size=(self.N_p, self.N_a))
        for i in range(self.N_p):
            for j in range(self.N_a):
                    if self.network[i, j] != 0:
                        beta_P[i, j] = beta_0[i, j] / np.sum(
                            self.network[i, :]
                        )**self.beta_trade_off

        with np.errstate(divide="ignore", invalid="ignore"):
            # return np.nan_to_num(R / (alpha_beta_A_prod)**self.q, copy=False, nan=0)
            beta_P = np.nan_to_num(beta_P / (self.alpha), copy=False, nan=0)

        beta_A = np.zeros((self.N_p, self.N_a))
        beta_0 = self.rng.uniform(low, high, size=(self.N_p, self.N_a))
        for i in range(self.N_p):
            for j in range(self.N_a):
                    if self.network[i, j] != 0:
                        # beta_A[i, j] = beta_0[i, j] / np.sum(
                        #     self.network[i, :]
                        # )**self.beta_trade_off
                        beta_A[i, j] = beta_0[i, j] / np.sum(
                            self.network[:, j]
                        )**self.beta_trade_off

        with np.errstate(divide="ignore", invalid="ignore"):
            # return np.nan_to_num(R / (alpha_beta_A_prod)**self.q, copy=False, nan=0)
            beta_A = np.nan_to_num(beta_A / (self.alpha), copy=False, nan=0)

        # beta_A = np.zeros((self.N_p, self.N_a))
        # beta_0 = self.rng.uniform(low, high, size=(self.N_p, self.N_a))
        # for i in range(self.N_p):
        #     for j in range(self.N_a):
        #             if self.network[i, j] != 0:
        #                 # beta_A[i, j] = beta_0[i, j] / np.sum(
        #                 #     self.network[i, :]
        #                 # )**self.beta_trade_off
        #                 beta_A[i, j] = beta_0[i, j]

        return beta_P, beta_A

    def _generate_C(self, low, high, diag_low, diag_high):
        """Generate competition matrices C

        Args:
            N_p [int]: number of plants
            N_a [int]: number of pollinators
            rng [np rng]: random number generator instance

        Returns:
            C_p [np array]: competition matrix plants
            C_a [np array]: competition matrix pollinators
        """
        C_p = self.rng.uniform(low, high, (self.N_p, self.N_p))
        np.fill_diagonal(C_p, self.rng.uniform(diag_low, diag_high, self.N_p))

        C_a = self.rng.uniform(low, high, (self.N_a, self.N_a))
        np.fill_diagonal(C_a, self.rng.uniform(diag_low, diag_high, self.N_a))

        return C_p, C_a

    def _phi(self, R, alpha_beta_A_prod):
        """Supply/demand-ratio. Use P = R if both are directly proportionate
        """
        # remove all possible nan values (divisions by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(R / (alpha_beta_A_prod)**self.q, copy=False, nan=0)
        # with np.errstate(divide="ignore", invalid="ignore"):
        #     return np.nan_to_num((R / alpha_beta_A_prod)**self.q, copy=False, nan=0)

    def _sample_params(self):

        # generate competition matrices
        # self.C_p, self.C_a = self._generate_C(0.025, 0.05, 0.9, 1.1)
        self.C_p, self.C_a = self._generate_C(0.01, 0.05, 0.8, 1.1)

        # mutualistic saturation factors h
        # self.h_p = self.rng.uniform(0.18, 0.22, self.N_p)
        # self.h_a = self.rng.uniform(0.18, 0.22, self.N_a)
        self.h_p = self.rng.uniform(0.15, 0.3, self.N_p)
        self.h_a = self.rng.uniform(0.15, 0.3, self.N_a)
        # self.h_p = self.rng.uniform(0.05, 0.2, self.N_p)
        # self.h_a = self.rng.uniform(0.05, 0.2, self.N_a)

        # intrinsic growth rates r
        self.r_p = self.rng.uniform(0.1, 0.35, self.N_p)
        self.r_a = self.rng.uniform(0.1, 0.35, self.N_a)
        # self.r_p = self.rng.uniform(0.05, 0.35, self.N_p)
        # self.r_a = self.rng.uniform(0.05, 0.35, self.N_a)

        self.alpha = self._generate_alpha()
        # self.alpha = self._generate_alpha_ones()

        # self.beta_P, self.beta_A = self._generate_beta(0.8, 1.2)
        self.beta_P, self.beta_A = self._generate_beta_normed(0.8, 1.2)


@numba.njit()
def _nu_term_numba(alpha, beta, N_p, N_a, nu):
    """Calculate nu term for alpha ODE.
    Numba significantly speeds up this computation.
    """
    nu_term = np.zeros((N_p, N_a))
    alpha_sum = alpha.sum(axis=0)
    for i in range(N_p):
        for j in range(N_a):
            if beta[i, j] != 0:
                nu_term[i, j] = 1/np.count_nonzero(beta[:, j]) - alpha[i, j]
    return nu_term * nu
