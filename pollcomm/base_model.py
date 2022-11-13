#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 17/04/2022
# ---------------------------------------------------------------------------
""" base_model.py

Implementation of Base model

"""
# ---------------------------------------------------------------------------
import copy
from timeit import default_timer as timer

import numpy as np

from pollcomm.ode_solver import solve_ode
from pollcomm.pollcomm_class import PollcommMutualismCompetition

__all__ = ["BaseModel"]


class BaseModel(PollcommMutualismCompetition):
    def __init__(
        self, N_p, N_a, mu=0, connectance=0.15, forbidden=0.3, nestedness=0.3,
        network_type="nested", rng=None, seed=None, gamma_trade_off=0.5, feasible=True
    ):
        print("Deprecated! Not guaranteed to work or give correct results...")
        t0 = timer()

        super().__init__(
            N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng, seed
        )

        self.gamma_trade_off = gamma_trade_off

        # if feasbile, sample parameters until we find a feasible solution for the
        # given network
        if feasible:
            feasible_iters = 100
            for i in range(feasible_iters):
                print(i)
                self._sample_params()
                sol = self.solve(100, n_steps=int(1e4))
                self.set_sol(sol)
                if self.is_all_alive()[-1]:
                    break
            else:
                print(f"No feasible network found in {feasible_iters} iterations")
        else:
            # sample parameters once and use these (don't check feasibility)
            self._sample_params()

        tf = timer()
        print(f"Created BaseModel instance. Time elapsed: {tf-t0:.5f} seconds\n")

    def __repr__(self):
        return "BM"

    def _sample_params(self):

        # generate competition matrices
        self.C_p, self.C_a = self._generate_C()

        # growth rates r, and mutualism factors h
        # print("# WARNING: r_p")
        self.r_p = self.rng.uniform(0.05, 0.35, self.N_p)
        # self.r_p = self.rng.uniform(0.2, 0.4, self.N_p)
        self.r_a = self.rng.uniform(0.05, 0.35, self.N_a)
        self.h_p = self.rng.uniform(0.15, 0.3, self.N_p)
        self.h_a = self.rng.uniform(0.15, 0.3, self.N_a)

        # generate gamma matrix
        self.gamma_P, self.gamma_A = self._generate_gamma()

    def ode(self, t, z, dA, C_p, C_a):
        """Full set of ODEs for pollcomm with fixed dA

        Returns:
            ODEs [np array]: full set of ODEs for pollcomm. Order: plants, pollinators
        """
        if isinstance(dA, (int, float)):
            _dA = dA
        elif isinstance(dA, (dict)):
            _dA = dA["func"](t, *dA.get("args", None))

        P = z[:self.N_p]
        A = z[self.N_p:]

        plant_ODEs = P * (
            self.r_p + self._mutualism(self.gamma_P @ A, self.h_p) -
            self._competition_vectorized(P, C_p)
        )  + self.mu

        poll_ODEs = A * (
            self.r_a - _dA + self._mutualism(self.gamma_A.T @ P, self.h_a) -
            self._competition_vectorized(A, C_a)
        ) + self.mu

        return np.concatenate((plant_ODEs, poll_ODEs))

    def solve(self, t_end, dA=0, dC=1, n_steps=int(1e5), y0=None):
        """Numerical solver of pollcomm ODEs. Makes use of solve_ivp() function of scipy

        Returns:
            sol [obj]: numerical solution of ODE in a scipy Bunch object
        """
        t_span = (0, t_end)

        # If no initial conditions are provided, use default initial conditions.
        if y0 is None:
            y0 = np.full(self.N, 1, dtype=float)

        C_p = dC * copy.deepcopy(self.C_p)
        C_a = dC * copy.deepcopy(self.C_a)

        t0 = timer()
        sol = solve_ode(
            self.ode, t_span, y0, n_steps, args=(dA, C_p, C_a)
        )
        print(f"Solved Base Model in {timer()-t0:.2f} seconds...\n")
        self.set_sol(sol)
        return sol

    def _generate_gamma(self):
        """Generate gamma (mutualism) matrix

        Returns:
            gamma [np array]: gamma mutualism matrix
        """
        gamma_P = np.zeros((self.N_p, self.N_a))
        gamma_0 = self.rng.uniform(0.8, 1.2, size=(self.N_p, self.N_a))
        for i in range(self.N_p):
            for j in range(self.N_a):
                    if self.network[i, j] != 0:
                        gamma_P[i, j] = gamma_0[i, j] / np.sum(
                            self.network[i, :]
                        )**self.gamma_trade_off

        gamma_A = np.zeros((self.N_p, self.N_a))
        gamma_0 = self.rng.uniform(0.8, 1.2, size=(self.N_p, self.N_a))
        for i in range(self.N_p):
            for j in range(self.N_a):
                    if self.network[i, j] != 0:
                        gamma_A[i, j] = gamma_0[i, j] / np.sum(
                            self.network[:, j]
                        )**self.gamma_trade_off

        return gamma_P, gamma_A

    def _generate_C(self):
        """Generate competition matrices C

        Args:
            N_p [int]: number of plants
            N_a [int]: number of pollinators
            rng [np rng]: random number generator instance

        Returns:
            C_p [np array]: competition matrix plants
            C_a [np array]: competition matrix pollinators
        """
        C_p = self.rng.uniform(0.01, 0.05, (self.N_p, self.N_p))
        np.fill_diagonal(C_p, self.rng.uniform(0.8, 1.1, self.N_p))

        C_a = self.rng.uniform(0.01, 0.05, (self.N_a, self.N_a))
        np.fill_diagonal(C_a, self.rng.uniform(0.8, 1.1, self.N_a))

        return C_p, C_a
