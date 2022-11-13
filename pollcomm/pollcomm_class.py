#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 24/01/2022
# ---------------------------------------------------------------------------
""" pollcomm.py

implementation in classes of the common base classes of the
plant-pollinator communities models

The absolute base class is PollcommBase implementing the mutualistic networks.
PollcommMutualismCompetition is the base class for the models with a Holling type II
mutualism and a competition term.

The different model implementations are:
    BaseModel: model as found in Lever et al. (2014)
    AdaptiveModel: similar to BaseModel, but includes adaptive foraging
    ValdovinosModel: model as found in Valdovinos et al. (2013)
"""
# ---------------------------------------------------------------------------
import copy
from timeit import default_timer as timer

import numpy as np

import pollcomm.networknw as nw
import pollcomm.stats as stats

__all__ = []


class PollcommBase():
    """Base class where the pollcomm bipartite nested networks are defined.
    This class implements the storage of the solution to the ODEs of the model
    (only latest calculated solution is kept).
    """
    def __init__(
        self, N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng, seed,
        *args, **kwargs
    ):
        self.seed = seed
        if rng is None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = rng

        self.N_p = N_p
        self.N_a = N_a
        self.N = self.N_p + self.N_a    # total population
        self.mu = mu                    # mortality rate
        self.connectance = connectance
        self.forbidden = forbidden
        self.nestedness = nestedness
        self.extinct_threshold = 0.01   # abundance below which species are extinct

        # create network
        # t0 = timer()
        if network_type == "scale_free":
            self.network, self.forbidden_network = nw.scale_free_network(
                self.N_p, 2, seed=self.seed
            )
        elif network_type == "nested":
            self.network, self.forbidden_network = nw.nested_network(
                self.N_p, self.N_a, self.connectance, self.forbidden, self.nestedness,
                self.rng
            )
        elif network_type == "random":
            self.network, self.forbidden_network = nw.nested_network(
                self.N_p, self.N_a, self.connectance, self.forbidden, self.nestedness,
                self.rng
            )
        elif isinstance(network_type, dict):
            # use an already generated network. Most of the times it is easier to
            # keep the same model instance and to just resample the parameters using
            # model.set_params()
            self.network = network_type["network"]
            self.forbidden_network = network_type["forbidden_network"]

            # recalculate statistics
            self.N_p = self.network.shape[0]
            self.N_a = self.network.shape[1]
            self.N = self.N_p + self.N_a
            self.connectance = self.network.sum() / (self.N_p * self.N_a)
            self.forbidden = self.forbidden_network.sum() / (self.N_p * self.N_a)
            self.nestedness = stats.nestedness_network(self.network)
        else:
            # load network from downloaded networks from weboflife database
            self.network, self.forbidden_network = nw.load_network_weboflife(
                network_type
            )

            # recalculate statistics
            self.N_p = self.network.shape[0]
            self.N_a = self.network.shape[1]
            self.N = self.N_p + self.N_a
            self.connectance = self.network.sum() / (self.N_p * self.N_a)
            self.forbidden = 0
            self.nestedness = stats.nestedness_network(self.network)

        # tf = timer()
        # print(f"\nGenerated network. Time elapsed: {tf-t0:.5f} seconds")

        # objects containing solution to ODEs
        self.reset_sol()

    def is_all_alive(self):
        """Checks if all species are alive at each timepoint"""
        if self.y_alive is None:
            return None
        return np.all(self.y_alive == 1, axis=0)

    def count_all_alive(self):
        """Counts how many species are alive at each timepoint"""
        return self.is_all_alive().count_nonzero()

    def reset_sol(self):
        """Remove all stored solutions. Should be called when same model is used with
        resampled parameters to prevent confusion of to which parameter set the solutions
        belong.
        """
        self.t = None           # time points of solution
        self.y = None           # solution
        self.t_partial = None   # time points of y_partial values
        self.y_partial = None   # solution of vars that are saved at certain time points
        self.y_all_end = None       # solution of all variables at last time point
        self.y_alive = None     # number of species alive per time point

    def set_sol(self, sol):
        """Set the solution from the ODEs and save them to instance variables"""
        self.t = sol.t
        self.y = sol.y
        self.set_species_alive(sol.y)

        if sol.y_partial is not None and len(sol.y_partial) > 0:
            self.t_partial = sol.t_partial
            self.y_partial = sol.y_partial
            self.y_all_end = np.concatenate((self.y[:, -1], self.y_partial[:, -1]))
        else:
            self.y_all_end = copy.deepcopy(self.y[:, -1])

    def set_species_alive(self, y=None):
        """Converts results of ODEs from abundance to species alive (1 alive, 0 extinct)
        """
        # solution needs to exist in order to use this function properly
        if y is None and self.y is None:
            return None

        y_alive = copy.deepcopy(y)
        y_alive[y_alive > self.extinct_threshold] = 1
        y_alive[y_alive <= self.extinct_threshold] = 0
        self.y_alive = y_alive.astype(int)


class PollcommMutualismCompetition(PollcommBase):
    """This class extends the PollcommBase class by including a few functions relevant
    for models implementing competition and mutualism functions.
    """
    def __init__(
        self, N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng, seed,
        *args, **kwargs
    ):
        super().__init__(
            N_p, N_a, mu, connectance, forbidden, nestedness, network_type, rng, seed
        )

    def _competition(self, N, C, i):
        """Competition term of pollcomm ODEs for a single species i in N

        Args:
            N [np array]: abundance of a group of species (plants or pollinators)
            C [2D np array]: competition matrix of species group (plants or pollinators)
            i [int]: index of N for which competition is calculated

        Returns:
            competition [int]: competition value for a single species i in N
        """
        return np.dot(C[i, :], N)

    def _competition_vectorized(self, N, C):
        """Competition term of pollcomm ODEs for all single species N

        Args:
            N [np array]: abundance of a group of species (plants or pollinators)
            C [2D np array]: competition matrix of species group (plants or pollinators)

        Returns:
            competition [np array]: competition array for species N
        """
        return C @ N

    def _mutualism(self, rho, h):
        """Holling type II mutualism function

        Args:
        rho [float/np array]: mutualism strength
        h [int]: h value for species of interest

        Returns:
            mutualism [float/np array]: mutualism value
        """
        return rho / (1 + h * rho)
