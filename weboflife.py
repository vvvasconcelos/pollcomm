#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 19/05/2022
# ---------------------------------------------------------------------------
""" weboflife.py

Calculate some statistics of the weboflife networks
"""
# ---------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt

import pollcomm as pc

# relative path to folder were networks are storeds
NETWORK_DB = "weboflife"

N_p = []
N_a = []
ratios = []
nestedness = []

# count the number of networks
count = 0

directory = os.fsencode(NETWORK_DB)
for file in os.listdir(directory):
     fname = os.fsdecode(file)
     if fname.endswith(".csv") and fname.startswith("M_PL_"):
         path = os.path.join(NETWORK_DB, fname)
         network = np.genfromtxt(path, delimiter=",")

         # count number of species
         N_p.append(network.shape[0])
         N_a.append(network.shape[1])
         ratios.append(network.shape[1]/network.shape[0])
         nestedness.append(pc.nestedness_network(network))

         count += 1

print(f"Calculated statistics over {count} networks.")

N_p = np.asarray(N_p)
N_a = np.asarray(N_a)
ratios = np.asarray(ratios)

print(f"Number of plant species: {np.mean(N_p):.2f} +- {np.std(N_p, ddof=1):.2f}.")
print(f"Number of pollinator species: {np.mean(N_a):.2f} +- {np.std(N_a, ddof=1):.2f}.")
print(f"Ratio pollinators over plants: {np.mean(ratios):.2f} +- {np.std(ratios, ddof=1):.2f}.")
