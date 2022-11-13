#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 23/01/2022
# ---------------------------------------------------------------------------
""" network.py

Network generation and manipulation
"""
# ---------------------------------------------------------------------------
import os
from timeit import default_timer as timer

import networkx as nx
import numpy as np

import pollcomm.stats as stats

__all__ = [
    "load_network_weboflife", "nested_network", "random_network",
    "scale_free_network", "sort_network"
]

NETWORK_DB = os.path.join("input", "web-of-life_2022-04-23_142330", "M_PL_")


def load_network_weboflife(id):
    """Load pollinator network from weboflife database

    Args:
        id [int]: network id, omitting M_PL_ (e.g., id='001')

    Returns:
        network [2D arr]: adjacency matrix bipartite network
        forbidden_network [2D arr]: zero matrix of bipartite network
    """
    try:
        fname = NETWORK_DB + id + ".csv"
        network = np.genfromtxt(fname, delimiter=",")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Network file does not exist, check path to input folder or try different id"
        )

    # convert network to binary
    network[network > 0] = 1

    # return network and an empty network of the same size (Forbidden Links network)
    # for compatibility purposes
    return network, np.zeros((network.shape[0], network.shape[1]))


def nested_network(
    N_p, N_a, connectance, forbidden, nestedness, rng, max_iter=int(1e5)
):
    """Generates a nested bipartite network (N_p by N_a)

    Args:
        N_p [int]: number of plant species
        N_a [int]: number of animal species
        connectance [float]: fraction of links (links/(N_p*N_a))
        forbidden [float]: fraction of forbidden links
        nestedness [float]: nestedness as fraction between 0 and 1
        rng [np rng]: rng instance
        Optional:
            max_iter [int]: maximum iterations to reach specified nestedness

    Returns:
        network [2D arr]: adjacency matrix bipartite network
        forbidden_network [2D arr]: forbidden links of bipartite network
    """
    # generate random network and network of forbidden interactions
    network, forbidden_network = random_network(
        N_p, N_a, connectance, forbidden, rng
    )

    iter = 0
    # t0 = timer()
    while stats.nestedness_network(network) < nestedness and iter < max_iter:
        iter += 1

        # randomly select one interaction between two species
        # select nonzero entries (the interactions)
        i, j = np.nonzero(network)
        ind = rng.choice(len(i))

        # the two species from this interaction
        a_plant, b_poll = i[ind], j[ind]

        # select third species
        c = rng.choice([0, 1])
        if c == 0:
            c_plant = rng.choice(N_p)

            # check if connection does not already exist
            if network[c_plant, b_poll] == 0 and a_plant != c_plant:

                # a_plant and c_plant should not be the same species
                # exception 1: a_plant only has one connection
                # exception 2: b_poll to c_plant is forbidden
                connections = np.sum(network[a_plant, :])
                if (
                    forbidden_network[c_plant, b_poll] == 0 and
                    connections > 1
                ):
                    # c_plant should have more interactions than a_plant
                    if (
                        np.sum(network[c_plant, :]) > connections
                    ):
                        network[c_plant, b_poll] = 1
                        network[a_plant, b_poll] = 0
        else:
            c_poll = rng.choice(N_a)

            # check if connection does not already exist
            if network[a_plant, c_poll] == 0 and b_poll != c_poll:

                # b_poll and c_poll should not be the same species
                # exception 1: b_poll only has one connection
                # exception 2: a_plant to c_poll is forbidden
                connections = np.sum(network[:, b_poll])
                if (
                    connections > 1 and
                    forbidden_network[a_plant, c_poll] == 0
                ):
                    # c_poll should have more interactions than b_poll
                    if (
                        np.sum(network[:, c_poll]) > connections
                    ):
                        network[a_plant, c_poll] = 1
                        network[a_plant, b_poll] = 0
    # tf = timer()
    # print(f"Time elapsed nested_network(): {tf-t0:.5f} seconds")

    # if max_iter reached, discard network and try again
    if iter == max_iter:
        print(
            "Network discarded, because it does not converge to given nestedness"
            "in max_iter time.",
            "Generating new network..."
        )
        return nested_network(
            N_p, N_a, connectance, forbidden, nestedness, rng, max_iter
        )

    # check if network is connected. If not, generate a new network
    if not _is_connected(network):
        print(
            "Network discarded, because it was not connected.",
            "Generating new network..."
        )
        return nested_network(
            N_p, N_a, connectance, forbidden, nestedness, rng, max_iter
        )

    # print("Network successfully generated!\n")
    return network, forbidden_network


def random_network(N_p, N_a, connectance, forbidden, rng, max_iter=1000):
    """Generates a random bipartite network (N_p by N_a)

    Args:
        N_p [int]: number of plant species
        N_a [int]: number of animal species
        connectance [float]: fraction of links (links/(N_p*N_a))
        forbidden [float]: fraction of forbidden links
        rng [np rng]: rng instance
        Optional:
            max_iter [int]: maximum tries to find feasible network

    Returns:
        network [2D arr]: adjacency matrix bipartite network
        forbidden_network [2D arr]: forbidden links of bipartite network
    """
    # continue generating new networks until a feasible one has been found or max_iter
    # reached
    for i in range(max_iter):
        network = np.zeros((N_p, N_a))
        forbidden_network = np.zeros((N_p, N_a))

        n_connected = int(connectance*N_p*N_a)
        n_forbidden = int(forbidden*N_p*N_a)

        # select indices for connections and forbidden links at the same time
        ind_selected = rng.choice(N_p*N_a, size=n_connected+n_forbidden, replace=False)

        # divide the selected indices over connected and forbidden links
        # according to their probabilities
        ind_connected = rng.choice(ind_selected, size=n_connected, replace=False)
        ind_forbidden = np.setdiff1d(ind_selected, ind_connected)

        # convert back to 2 dimensional indices
        ind_connected = np.column_stack(np.unravel_index(ind_connected, (N_p, N_a)))
        ind_forbidden = np.column_stack(np.unravel_index(ind_forbidden, (N_p, N_a)))

        # put connection into adjacency matrix
        network[tuple(ind_connected.T)] = 1

        # save forbidden links into a matrix
        forbidden_network[tuple(ind_forbidden.T)] = 1

        # check if all species have at least one interaction
        if (
            not (~network.any(axis=0)).any() and
            not (~network.any(axis=1)).any()
        ):
            return network, forbidden_network
    else:
        # no random network found. Try to generate a new one
        print(
            "Network discarded, because not all species have interactions.",
            "Generating new network..."
        )
        return random_network(N_p, N_a, connectance, forbidden, rng, max_iter)


def scale_free_network(n, m, seed=None):
    """Generate scale-free network and return adjacency matrix as numpy array
    rows are plants, columns are pollinators

    Args:
        n [int]: number of nodes
        m [int]: number of initial edges between two nodes
        Optional:
            seed [int]: seed or instance of rng

    Returns:
        network [2D arr]: adjacency matrix
        forbidden_network [2D arr]: zero matrix for consistency with output other
            network generators
    """
    return nx.convert_matrix.to_numpy_array(
        nx.generators.random_graphs.barabasi_albert_graph(n, m, seed=seed)
    ), np.zeros((n, n))


def sort_network(network, *args):
    """Sort network from most connected to least connected elements

    Args:
        network [2D arr]: adjacency matrix
        Optional:
            args [2D arr]: networks to sort accordingly to network

    Returns:
        network [2D arr]: sorted adjacency matrix
        Optional:
            args [2D arr]: sorted (if supplied to function)
    """
    inds1 = np.argsort(network.sum(axis=0))
    network = network[:, inds1[::-1]]

    inds2 = np.argsort(network.sum(axis=1))
    network = network[inds2[::-1], :]

    if args:
        sorted_args = []
        for arg in args:
            arg = arg[:, inds1[::-1]]
            arg = arg[inds2[::-1], :]
            sorted_args.append(arg)

        return (network, ) + tuple(arg for arg in sorted_args)
    return network


def _bfs(network):
    """Breadth-first search for finding all connected nodes starting from 0 in a
    bipartite network

    Adapted-to work for bipartite networks-from:
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/components/
    connected.html

    Args:
        network [2D arr]: adjacency matrix bipartite network

    Returns:
        seen [set]: set of all nodes reached from 0 node

    """
    N_p, N_a = network.shape
    seen = set()
    nextlevel = {0}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for node in thislevel:
            if node not in seen:
                seen.add(node)

                # find all connected nodes to node
                if node < N_p:
                    for j in range(N_a):
                        if network[node, j] > 0:
                            nextlevel.add(N_p + j)
                else:
                    for i in range(N_p):
                        if network[i, node-N_p] > 0:
                            nextlevel.add(i)
    return seen


def _is_connected(network):
    """Checks if all nodes in a bipartite network are connected

    Args:
        network [2D arr]: adjacency matrix bipartite network

    Returns:
        ... [bool]: True if connected, else False
    """
    return np.sum([1 for node in _bfs(network)]) == np.sum(network.shape)
