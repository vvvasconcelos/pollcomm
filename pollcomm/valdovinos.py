#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 08/04/2022
# ---------------------------------------------------------------------------
""" valdovinos.py

Implementation of Valdovinos model (Valdovinos et al. 2013)

"""
# ---------------------------------------------------------------------------
import copy
from timeit import default_timer as timer

import numpy as np

from pollcomm.ode_solver import solve_ode
from pollcomm.pollcomm_class import PollcommBase

__all__ = ["ValdovinosModel"]


def uniform_var(rng, mean, var, size=None):
    low = mean - mean * var
    high = mean + mean * var
    return rng.uniform(low, high, size)


class ValdovinosModel(PollcommBase):
    def __init__(
        self, N_p, N_a, mu=0, connectance=0.15, forbidden=0.3, nestedness=0.3,
        network_type="nested", rng=None, seed=None
    ):
        super().__init__(
            N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng, seed
        )

        # create initial alpha by normalizing network
        self.alpha0 = self.network / self.network.sum(axis=0)

        self.p_var = 0.1     # parameter variance for plants
        self.a_var = 0.0001  # parameter variance for pollinators

        # fixed parameters
        self.tau = uniform_var(self.rng, 1, self.a_var, size=(self.N_p, self.N_a))
        self.e = uniform_var(self.rng, 0.8, self.p_var, size=(self.N_p, self.N_a))
        self.mu_p = uniform_var(self.rng, 0.002, self.p_var, size=self.N_p)
        self.mu_a = uniform_var(self.rng, 0.01, self.a_var, size=self.N_a)
        self.c = uniform_var(self.rng, 0.2, self.a_var, size=(self.N_p, self.N_a))
        self.b = uniform_var(self.rng, 0.4, self.a_var, size=(self.N_p, self.N_a))
        self.b_max = uniform_var(self.rng, 0.4, self.a_var, size=(self.N_p, self.N_a))
        self.kappa = uniform_var(self.rng, 0.4, self.a_var, size=(self.N_p, self.N_a))
        self.g = uniform_var(self.rng, 0.4, self.p_var, size=self.N_p)
        self.u = uniform_var(self.rng, 0.002, self.p_var, size=self.N_p)
        self.w = uniform_var(self.rng, 1.2, self.p_var, size=self.N_p)
        self.beta = uniform_var(self.rng, 0.2, self.p_var, size=self.N_p)
        self.phi = uniform_var(self.rng, 0.04, self.p_var, size=self.N_p)
        self.G = uniform_var(self.rng, 2, self.a_var, size=self.N_a)
        self.K = uniform_var(self.rng, 20, self.a_var, size=self.N_a)

    def __repr__(self):
        return "VM"

    def LFR(self, R, P, i, j):
        return self.b[i, j] * R / P

    def LFR_vectorized(self, R, P):
        # self.b[i, j] * R / P
        return (self.b.T * (R / P)).T

    def NFR(self, R, P, i, j):
        return self.b_max[i, j] * (R[i] / (self.kappa[i, j] * P[i] + R[i]))

    def NFR_vectorized(self, R, P):
        # self.b_max[i, j] * (R[i] / (self.kappa[i, j] * P[i] + R))
        return self.b_max * (R / (self.kappa.T * P + R)).T

    def ode(self, t, z, dA):
        """Full set of ODEs

        Returns:
            ODEs [np array]: full set of ODEs
        """
        if isinstance(dA, (int, float)):
            _dA = dA
        elif isinstance(dA, (dict)):
            _dA = dA["func"](t, *dA.get("args", None))

        # unpack state variables
        P = z[:self.N_p]
        A = z[self.N_p:self.N]
        R = z[self.N:self.N+self.N_p]
        alpha = z[self.N+self.N_p:].reshape((self.N_p, self.N_a))

        F = self.LFR_vectorized(R, P)

        V = ((alpha * self.tau).T * P).T * A + self.mu

        sigma = V / V.sum(axis=0)

        gamma = self.g * (1 - self.u @ P - (self.w - self.u) * P)

        # calculate the differential equations
        plant_ODEs = gamma * (self.e * sigma * V).sum(axis=1) - (self.mu_p) * P

        poll_ODEs = (self.c * V * F).sum(axis=0) - (self.mu_a + _dA) * A

        R_ODEs = self.beta * P - self.phi * R - (V * F).sum(axis=1)

        alpha_ODEs = (self.G * alpha * (
            ((self.c * self.tau * F).T * P).T -\
            (self.tau.T * P).T * (alpha * self.c * F).sum(axis=0)
        )).flatten()

        return np.concatenate((plant_ODEs, poll_ODEs, R_ODEs, alpha_ODEs))

    def solve(self, t_end, dA=0, n_steps=int(1e6), y0=None, save_period=0):
        """Numerical solver of ODEs. Makes use of solve_ivp() function of scipy

        Returns:
            sol [obj]: numerical solution of ODE in a scipy Bunch object
        """
        t_span = (0, t_end)

        # If no initial conditions are provided, use default initial conditions.
        if y0 is None:
            y0 = uniform_var(self.rng, 0.5, self.p_var, self.N_p)
            y0 = np.concatenate((y0, uniform_var(self.rng, 0.5, self.p_var, self.N_a)))
            y0 = np.concatenate((y0, uniform_var(self.rng, 0.5, self.p_var, self.N_p)))
            alpha = copy.deepcopy(self.alpha0)
            y0 = np.concatenate((y0, alpha.flatten()))

        save_partial = {
            "ind": (self.N + self.N_p, y0.shape[0]-1),
            "save_period": save_period
        }

        t0 = timer()
        sol = solve_ode(
            self.ode, t_span, y0, n_steps, args=(dA,), save_partial=save_partial
        )
        print(f"Solved Valdovinos Model in {timer()-t0:.2f} seconds...\n")
        self.set_sol(sol)
        return  sol
