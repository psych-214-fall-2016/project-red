"""
Segmentation using k-means clustering. Each pixel is assigned to the cluster
with the closest mean.

@author: Christine Tseng
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def should_go(centers, old_centers, iteration, max_iter):
    """
    Returns whether kmeans should run for another iteration.

    Input
    -----
    centers : array (k,)
        Contains the means for each cluster after the newest iteration.

    old_centers : array (k,)
        Contains the means for each cluster from the previous iteration.

    iteration : int
        Current iteration number.

    max_iter : int
        Maximum number of iterations.

    Output
    ------
    - : boolean
        True if kmeans should centers != old_centers and iteration < max_iter.
        False otherwise.
    """

    if iteration > max_iter:
        return False
    return not np.allclose(centers, old_centers)

def get_labels(x, centers):
    """
    Returns labels/cluster assignments for each pixel given the cluster means.
    Each pixel is assigned to the cluster that has the closest mean.

    Input
    -----
    x : array (n,)
        Input data.

    centers : array (k,)
        Contains the means for each cluster.

    Output
    ------
    labels : array (n,)
        Contains the cluster assignment for each pixel.
    """
    labels = np.zeros(x.shape)
    for i, xpt in enumerate(x):
        dist = np.abs(centers - xpt)
        labels[i] = np.argmin(dist)
    return labels

def get_centers(x, labels, k):
    """
    Return means for clusters given cluster assignments for pixels. Each cluster
    mean is updated to be the mean of the pixels in the cluster.

    Input
    -----
    x : array (n,)
        Input data.

    labels : array (n,)
        Contains the cluster assignment for each pixel.

    k : int
        Number of clusters.

    Output
    ------
    centers : array (k,)
        Contains the updated means for each cluster.
    """
    centers = np.zeros(k)
    for i in range(k):
        x_cluster = x[labels == i]
        centers[i] = x_cluster.mean() # update
    centers[np.isnan(centers)] = 0 # avoid nans
    return centers

def kmeans(x, k=3, max_iter=10^4, scale_max=50, scale_min=0,
           map_keys=['csf', 'gray', 'white']):
    """
    Run kmeans for data x and probability maps for each cluster.

    Input
    -----
    x : array (n,)
        Input data.

    k : int, default=3
        Number of clusters. Usually k=3 for white matter, gray matter, and csf.

    max_iter : int, default=10^4
        Maximum number of k-means iterations.

    scale_max, scale_min : ints, defaults=50, 0
        Initial guesses for cluster means are chosen from a uniform distribution
        between scale_min and scale_max.

    map_keys : list of strings
        Names for each cluster.

    Output
    ------
    centers : array (k,)
        Contains the updated means for each cluster.

    labels : array (n,)
        Contains the cluster assignment for each pixel.

    maps : dict (k,)
        Contains the probability map for each cluster.
    """
    centers = np.random.rand(k) * (scale_max - scale_min) + scale_min
    old_centers = np.zeros(k)
    iteration = 0

    # Find best clustering
    while should_go(centers, old_centers, iteration, max_iter):
        old_centers = centers
        iteration += 1
        labels = get_labels(x, centers)
        centers = get_centers(x, labels, k)

    # Generate maps - each pixel has probability = 1 of being in its cluster
    # and 0 in the other clusters
    maps = dict()
    sort_idx = np.argsort(centers)
    for i, k in enumerate(sort_idx):
        m = np.zeros(len(labels))
        m[np.argwhere(labels == k)] = 1
        maps[map_keys[i]] = m

    return centers, labels, maps
