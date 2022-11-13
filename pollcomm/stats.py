#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 19/01/2022
# ---------------------------------------------------------------------------
""" stat.py

This module contains functions to calculate the nestedness of bipartite networks
"""
# ---------------------------------------------------------------------------
import numpy as np
import numba

__all__ = ["nestedness_network"]


@numba.njit()
def nestedness_network(pollcomm_network):
    """Calculate nestedness from adjacancy matrix of pollcomm network"""
    N_p, N_a = pollcomm_network.shape

    nestedness_plant = _nestedness_plant_species(pollcomm_network)
    nestedness_polls = _nestedness_poll_species(pollcomm_network)
    return (nestedness_plant + nestedness_polls) / (N_p*(N_p-1)/2 + N_a*(N_a-1)/2)


@numba.njit()
def _nestedness_plant_species(pollcomm_network):
    N_p, N_a = pollcomm_network.shape
    nestedness = 0

    # Nestedness of plant species
    for i in range(N_p-1):
        for j in range(i+1, N_p):
            n_i = np.sum(pollcomm_network[i])
            n_j = np.sum(pollcomm_network[j])
            n_ij = 0
            for k in range(N_a):
                if pollcomm_network[i, k] == 1 and pollcomm_network[j, k] == 1:
                    n_ij += 1

            # prevent dividing by zero
            if np.amin(np.array([n_i, n_j])) > 0.:

                nestedness += n_ij / np.amin(np.array([n_i, n_j]))

    return nestedness


@numba.njit()
def _nestedness_poll_species(pollcomm_network):
    N_p, N_a = pollcomm_network.shape
    nestedness = 0

    # Nestedness of plant species
    for i in range(N_a-1):
        for j in range(i+1, N_a):
            n_i = np.sum(pollcomm_network[:, i])
            n_j = np.sum(pollcomm_network[:, j])
            n_ij = 0
            for k in range(N_p):
                if pollcomm_network[k, i] == 1 and pollcomm_network[k, j] == 1:
                    n_ij += 1

            # prevent dividing by zero
            if np.amin(np.array([n_i, n_j])) > 0.:
                nestedness += n_ij / np.amin(np.array([n_i, n_j]))

    return nestedness
