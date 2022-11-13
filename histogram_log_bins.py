# Credits: Fundamental Topics in Statistical Physics 1 @ UvA
# Author: Karina Gonzalez Lopez
#

import numpy as np


def histogram_log_bins(x, x_min=None, x_max=None, num_of_bins=20, min_hits=1):
    """
    Generate histogram of x with logarithmically spaced bins.
    """
    if not x_min:
        x_min = np.min(x)
    if not x_max:
        x_max = np.max(x)

    # This is the factor that each subsequent bin is larger than the next.
    growth_factor = (x_max/x_min) ** (1/(num_of_bins + 1))
    # Generates logarithmically spaced points from x_min to x_max.
    bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), num=num_of_bins+1)
    # We don't need the second argument (which are again the bin edges).
    # It's conventional to denote arguments you don't intend to use with _.

    bin_counts,_ = np.histogram(x, bins=bin_edges)
    total_hits = np.sum(bin_counts)
    bin_counts = bin_counts.astype(float)

    # Rescale bin counts by their relative sizes.
    significant_bins = []
    for bin_index in range(np.size(bin_counts)):
        if bin_counts[bin_index] >= min_hits:
            significant_bins.append(bin_index)

        bin_counts[bin_index] = bin_counts[bin_index] / (growth_factor**bin_index)

    # Is there a better way to get the center of a bin on logarithmic axis?
    # There probably is, please figure it out.
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # You can optionally rescale the counts by total_hits if you want to get a density.
    return bin_counts[significant_bins], bin_centers[significant_bins], total_hits
