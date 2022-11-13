#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 14/05/2022
# ---------------------------------------------------------------------------
""" adaptive_resource_model.py

Implementation of Adaptive Resource Model

Deprecated, not guaranteed to work

"""
# ---------------------------------------------------------------------------
import copy
from timeit import default_timer as timer

import numba
import numpy as np

from pollcomm.ode_solver import solve_ode
from pollcomm.adaptive_model import AdaptiveModel

__all__ = ["AdaptiveResourceModel"]


class AdaptiveResourceModel(AdaptiveModel):
    def __init__(
        self, N_p, N_a, mu=0, connectance=0.15, forbidden=0.3, nestedness=0.3,
        network_type="nested", rng=None, seed=None, nu=0.1, G=1, q=1, feasible=False
    ):
        print("Deprecated! Not guaranteed to work or give correct results...")
        t0 = timer()

        self.theta = 1
        self.kappa = 5

        super().__init__(
        N_p, N_a, mu=mu, connectance=connectance, forbidden=forbidden,
        nestedness=nestedness, network_type=network_type, rng=rng, seed=seed, nu=nu,
        G=G, q=q, feasible=feasible
        )

        # print("# WARNING: multiplying h_a in ARM")
        # self.h_a = 3*self.rng.uniform(0.25, 0.3, self.N_a)

        tf = timer()
        print(f"Created AdaptiveResourceModel instance. Time elapsed: {tf-t0:.5f} seconds\n")

    def __repr__(self):
        return "ARM"

    def ode(self, t, z, dA):
        """Full set of ODEs for pollcomm with fixed dA

        Returns:
            ODEs [np array]: full set of ODEs for pollcomm. Order: plants, pollinators
        """
        if isinstance(dA, (int, float)):
            _dA = dA
        elif isinstance(dA, (dict)):
            _dA = dA["func"](t, *dA.get("args", None))

        P = z[:self.N_p]
        A = z[self.N_p:self.N]
        alpha = z[self.N:self.N + self.N_p * self.N_a].reshape((self.N_p, self.N_a))
        R = z[self.N + self.N_p * self.N_a:]

        # R should be equal or greater than zero
        # should not be necessary, unless solver has some error
        R[R < 0] = 0

        # precalculating these term, since they are used more than once
        alpha_beta_A_prod = self._alpha_beta_A_prod(alpha, A)
        alpha_beta_R_prod = self._alpha_beta_R_prod(alpha, R)
        # alpha_beta_phi_prod = self._alpha_beta_phi_prod(alpha, R, alpha_beta_A_prod)

        # P_alpha_beta_A_prod = self._alpha_beta_A_prod((alpha.T/alpha.sum(axis=1)).T, A)
        plant_ODEs = P * (
            self.r_p + self._mutualism(alpha_beta_A_prod, self.h_p) -
            self._competition_vectorized(P, self.C_p)
        )  + self.mu

        poll_ODEs = A * (
            self.r_a - _dA + self._mutualism(alpha_beta_R_prod, self.h_a) -
            self._competition_vectorized(A, self.C_a)
        ) + self.mu

        alpha_ODEs = self.G * (alpha *
            (
                self.beta *
                R[:, np.newaxis] -
                alpha_beta_R_prod
            ) + _nu_term_numba(alpha, self.beta, self.N_p, self.N_a, self.nu)
        )

        R_ODEs = self.theta * P - (self.theta / self.kappa) * R - alpha_beta_A_prod * R

        return np.concatenate((plant_ODEs, poll_ODEs, alpha_ODEs.flatten(), R_ODEs))

    def solve(
        self, t_end, dA=0, n_steps=int(1e5), y0=None, save_period=0,
        stop_on_collapse=False, extinct_threshold=None
    ):
        """Numerical solver of pollcomm ODEs. Makes use of solve_ivp() function of scipy

        Returns:
            sol [obj]: numerical solution of ODE in a scipy Bunch object
        """
        t_span = (0, t_end)

        # If no initial conditions are provided, use default initial conditions.
        if y0 is None:
            y0 = np.full(self.N, 0.5, dtype=float)
            y0 = np.concatenate((y0, self.alpha.flatten()))
            y0 = np.concatenate((y0, np.full(self.N_p, 0.5, dtype=float)))

        save_partial = {
            "ind": (self.N, self.N+len(self.alpha.flatten())-1),
            "save_period": save_period
        }

        t0 = timer()
        print("Solving...")
        sol = solve_ode(
            self.ode, t_span, y0, n_steps, args=(dA,), save_partial=save_partial,
            rtol=1e-3, atol=1e-6, method="LSODA", stop_on_collapse=stop_on_collapse,
            N_p=self.N_p, N_a=self.N_a, extinct_threshold=extinct_threshold
        )
        print(f"Solved Adaptive Model in {timer()-t0:.2f} seconds...\n")
        self.set_sol(sol)
        return  sol

    def _alpha_beta_R_prod(self, alpha, R):
        """Calculates np.sum(
            [alpha[i, j] * self.beta[i, j] * R[i] for i in range(self.N_p)]
        ) in a vectorized manner for all j in N_a
        """
        return (alpha * self.beta).T @ R
