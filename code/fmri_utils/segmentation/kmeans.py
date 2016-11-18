import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def should_go(centers, old_centers, iteration, max_iter):
    if iteration > max_iter
    return not np.all_close(centers, old_centers)

def get_labels(x, centers):
    labels = np.zeros(x.shape)
    for i, xpt in enumerate(x):
        dist = np.abs(centers - xpt)
        labels[i] = np.argmin(dist)
    return labels

def get_centers(x, labels, k):
    centers = np.zeros(k)
    for i in range(k):
        x_cluster = x[labels == i]
        centers[i] = x_cluster.mean()
    return centers

def kmeans(x, k=4, max_iter=10^4):
    centers = np.random.rand(k)
    old_centers = None
    iteration = 0

    while should_go(centers, old_centers, iteration, max_iter):
        old_centers = centers
        iteration += 1
        labels = get_labels(x, centers)
        centers = get_centers(x, labels, k)
    return centers
