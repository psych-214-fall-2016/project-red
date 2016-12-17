"""
Segmentation using a markov random field model and expectation-maximization.
I tried to implement FSL's FAST segmentation, which is based off this paper:

Zhang, Y. and Brady, M. and Smith, S. Segmentation of brain MR images through a
hidden Markov random field model and the expectation-maximization algorithm.
IEEE Trans Med Imag, 20(1):45-57, 2001.

@author: Christine Tseng
"""

import numpy as np
import nibabel as nib
import copy
import scipy.misc
import scipy.ndimage as ndimage
from joblib import Parallel, delayed
from fmri_utils.segmentation.kmeans import kmeans


### FUNCTIONS FOR MARKOV RANDOM FIELD MODEL ###

def log_likelihood(thetas, data, labels):
    """
    Returns U(y|x), which is proportional to log(P(y|x)). Here y is the set of
    pixel labels, x the set of pixel intensities, and
    U(y|x) = \sum_{i \in S} [(y_i - \mu_{x_i})^2 / (2 * \sigma_{x_i}^2)
                            + log(\sigma_{x_i})]

    Input
    -----
    thetas : list of lists
        Contains [mu, sigma] for each label.

    data : array (n, m)
        Image to segment.

    labels : int array (n, m)
        Label for each pixel.

    Output
    ------
    u : scalar
        U(y|x).
    """
    u = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            label_idx = labels[i, j]
            mu, sigma = thetas[label_idx]
            u += (1 / (2 * sigma**2)) * (data[i, j] - mu)**2 + np.log(sigma)
    return u

def log_prior(pairs, labels, beta):
    """
    Returns U(x), which is proportional to log(P(x)). Here x is the set of
    pixel intensities.

    Input
    -----
    pairs : list of lists
        Each list contains pair [[x1, y1], [x2, y2]] of pixels where [x1, y1]
        and [x2, y2] are neighbors.

    labels : int array (n, m)
        Labels for each pixel.

    beta : float
        Parameter for determining how much two neighboring pixels should be
        forced to have the same label. Higher beta values correspond with more
        homogeneity.

    Output
    ------
    u : float
        U(x).
    """
    u = 0
    for pair in pairs:
        x1, y1 = pair[0]
        x2, y2 = pair[1]
        # Increase prior if neighboring pixels have same labeling
        if labels[x1, y1] == labels[x2, y2]:
            u += beta
    return u

def get_pairs(data):
    """
    Returns locations of unique pairs of neighboring pixels in data.

    Input
    -----
    data : array (n, m)
        Image to segment.

    Output
    ------
    pairs : list of lists
        Each list contains pair [[x1, y1], [x2, y2]] of pixels where [x1, y1]
        and [x2, y2] are neighbors.
    """
    pairs = []
    # Iterate over all pixels to get neighbors
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if i != data.shape[0]-1:
                pairs.append([[i, j], [i+1, j]]) # south
            if (i != data.shape[0]-1) and (j != data.shape[1]-1):
                pairs.append([[i, j], [i+1, j+1]]) # southeast
            if (j != 0) and (i != data.shape[0]-1):
                pairs.append([[i, j], [i+1, j-1]]) # southwest
            if (j != data.shape[1]-1):
                pairs.append([[i, j], [i, j+1]]) # east
    return pairs

def log_posterior(data, thetas, labels, beta, pairs):
    """
    Returns U(x|y) = U(y|x) + U(x). log P(y|x) is proportional to -U(x|y).

    Input
    -----
    data : array (n, m)
        Image to segment.

    thetas : list of lists
        Contains [mu, sigma] for each label.

    labels : int array (n, m)
        Labels for each pixel.

    beta : float
        Parameter for determining how much two neighboring pixels should be
        forced to have the same label. Higher beta values correspond with more
        homogeneity.

    Output
    ------
    posterior : float
        U(x|y).
    """
    likelihood = np.sum(log_likelihood(thetas, data, labels))
    prior = log_prior(pairs, labels, beta)
    posterior = likelihood + prior
    return posterior

def should_go(label1, label2, iteration, max_iter):
    """
    Returns whether ICM should run for another iteration.

    Input
    -----
    label1 : int array (n, m)
        First set of labels.

    label2 : int array (n, m)
        Second set of labels.

    iteration : int
        Number of ICM iterations completed so far.

    max_iter : int
        Max number of ICM iterations.

    Output
    ------
    - : boolean
        False if label1 == label2 or iter > max_iter, True otherwise.
    """
    if iteration > max_iter:
        return False
    return not np.allclose(label1, label2)

def get_best_label(p, data, thetas, labels, L, beta, pairs):
    """
    Returns the best labeling over the len(L) labelings possible by changing the
    label of pixel p.

    Inputs
    ------
    p : int
        Index of pixel in terms of the unraveled labels matrix.

    data : array (n, m)
        Image to segment.

    thetas : list of lists
        Contains [mu, sigma] for each label.

    labels : int array (n, m)
        Labels for each pixel.

    L : int array (# labels, )
        Array containing possible labels.

    beta : float
        Parameter for determining how much two neighboring pixels should be
        forced to have the same label. Higher beta values correspond with more
        homogeneity.

    pairs : list of lists
        Each list contains pair [[x1, y1], [x2, y2]] of pixels where [x1, y1]
        and [x2, y2] are neighbors.

    Output
    ------
    best_label : int
        Best label for pixel p.
    """
    min_energy = np.inf
    best_label = None
    labels_r = labels.ravel()

    for label in L:
        # Make new labeling
        labels_changed = copy.copy(labels_r)
        labels_changed[p] = label
        labels_changed = labels_changed.reshape(labels.shape)
        # Compute energy of new labeling
        post_energy = log_posterior(data, thetas, labels_changed, beta, pairs)
        # Update if new labeling has less energy
        if post_energy < min_energy:
            min_energy = post_energy
            best_label = label
    return best_label, min_energy

def get_labels(data, thetas, current_labels, L, beta, max_iter, njobs):
    """
    Do Iterative conditional modes (ICM) to find next best set of labels given
    the thetas. Returns the best labeling.

    Input
    -----
    data : array (n, m)
        Image to segment.

    thetas : list of lists
        Contains [mu, sigma] for each label.

    current_labels : int arrray (n, m)
        Current best labeling for image.

    L : int array (# labels, )
        Array containing possible labels.

    beta : float
        Parameter for determining how much two neighboring pixels should be
        forced to have the same label. Higher beta values correspond with more
        homogeneity.

    max_iter : int
        Maximum number of ICM iterations.

    njobs : int
        Number of jobs to do in parallel.

    Output
    ------
    current_labels : int array (n, m)
        Best labeling for image.
    """
    prev_labels = np.ones(data.shape) * -1
    pairs = get_pairs(data)

    # Iterate through all possible labelings that have a difference of at most
    # one label and find the one that minimizes the posterior.
    iteration = 0
    while should_go(prev_labels, current_labels, iteration, max_iter):
        # Estimate best label for each pixel in parallel
        new_labels = Parallel(n_jobs=njobs, backend='threading')(delayed(get_best_label)(
                p, data, thetas, current_labels, L, beta, pairs) for p in range(data.size))
        p_labels, p_min_energy = zip(*new_labels)
        # Find best labeling across pixels
        best_labels = copy.copy(current_labels).ravel()
        best_idx = np.argmin(p_min_energy)
        best_labels[best_idx] = p_labels[best_idx]
        # Update
        prev_labels = current_labels
        current_labels = best_labels.reshape(current_labels.shape)
        iteration += 1
    return current_labels


### FUNCTIONS FOR EXPECTATION MAXIMIZATION ###

def likelihood(data, theta):
    """
    Returns p(y|mu_l, sigma_l) = Gaussian(mu_l, sigma_l). y is the set of pixel
    intensities and mu_l, sigma_l are the mean and standard deviation for label l.
    This gives the likelihood that pixel p should be labeled with l given only
    p's intensity.

    Input
    -----
    data : array (n, m)
        Image to segment.

    theta : list
        Contains [mu, sigma] for label l.

    Output
    ------
    - : float
        p(y|mu_l, sigma_l)
    """
    mu, sigma = theta
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(data - mu)**2 / (2 * sigma**2))

def get_neighbors(loc, dshape):
    """
    Get immediate neighbors of location loc given matrix of shape dshape.

    Input
    -----
    loc : list
        Contains [row, column] of location of interest in image.

    dshape : tuple
        Shape of image.

    Output
    ------
    neighbors : list of lists
        Each list contains [row, column] of a neighboring pixel.
    """
    neighbors = []
    x, y = loc
    if x > 0: neighbors.append([x-1, y]) # north
    if x + 1 < dshape[0]: neighbors.append([x+1, y]) # south
    if y > 0: neighbors.append([x, y-1]) # west
    if y + 1 < dshape[1]: neighbors.append([x, y+1]) # east
    if (x > 0) and (y > 0): neighbors.append([x-1, y-1]) # northwest
    if (x > 0) and (y+1 < dshape[1]): neighbors.append([x-1, y+1]) # northeast
    if (x+1 < dshape[0]) and (y > 0): neighbors.append([x+1, y-1]) # southwest
    if (x+1 < dshape[0]) and (y+1 < dshape[1]): neighbors.append([x+1, y+1]) # southeast
    return neighbors

def p_label(label, L, beta, loc, current_labels):
    """
    Get the probability of a pixel at location loc being labeled with label given
    the labels of its neighbors.

    Input
    -----
    label : int
        Label being considered.

    L : int array (# labels, )
        Array containing possible labels.

    beta : float
        Parameter for determining how much two neighboring pixels should be
        forced to have the same label. Higher beta values correspond with more
        homogeneity.

    loc : list
        Contains [row, column] of pixel of interest in image.

    current_labels : int array (n, m)
        Current best labeling of image.

    Output
    ------
    - : float
        Probability of pixel being labeled with label.
    """
    x, y = loc
    neighbors = get_neighbors(loc, current_labels.shape)

    # Normalization factor
    norm = 0
    n_neighbors = len(neighbors)
    for i in range(n_neighbors+1):
        norm += scipy.misc.comb(n_neighbors, i)

    # Find probability
    p_labels = np.zeros(len(L))
    for n in neighbors:
        x2, y2 = n
        for l in L:
            if l == current_labels[x2, y2]:
                p_labels[l] += beta
    Z = np.sum(np.exp(-p_labels))
    p = (1 / Z) * np.exp(-p_labels[label])
    return p

def update_thetas(data, thetas, beta, L, current_labels):
    """
    Update centers and variances of estimated Gaussians for labels.

    Input
    -----
    data : array (n, m)
        Image to segment.

    thetas : list of lists
        Contains [mu, sigma] for each label.

    beta : float
        Parameter for determining how much two neighboring pixels should be
        forced to have the same label. Higher beta values correspond with more
        homogeneity.

    L : int array (# labels, )
        Array containing possible labels.

    current_labels: int array (n, m)
        Current labeling of image.

    Output
    ------
    new_thetas : list of lists
        Contains updated [mu, sigma] for each label.
    """
    new_thetas = []

    for l, theta in enumerate(thetas):
        num_mu = num_sigma = denom = 0
        label = l
        g = likelihood(data, theta)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                p_lx = p_label(label, L, beta, [i, j], current_labels)
                # Update sums for each pixel
                num_mu += g[i, j] * p_lx * data[i, j]
                num_sigma += g[i, j] * p_lx * (data[i, j] - theta[0])**2
                denom += g[i, j] * p_lx
        new_mu = num_mu / denom
        new_sigma = np.sqrt(num_sigma / denom)
        new_thetas.append([new_mu, new_sigma])

    return new_thetas


### MAIN FUNCTIONS ###
def init_values(data, k, scale_range, scale_sigma):
    """
    Initialize theta and labels. The means of each label are initialized using
    kmeans. The standard deviation of each label is chosen randomly from a uniform
    distribution between 0 and scale_sigma.

    Input
    -----
    data : array (n, m)
        Image to segment.

    k : int
        Number of labels.

    scale_range : tuple
        (min, max) Range of values from which start kmeans.

    scale_sigma : float
        Sigma will be initialized with a random value between 1 and scale_sigma.

    Output
    ------
    thetas : list of lists
        Contains [mu, sigma] for each label.

    labels : int array (n, m)
        Labels for each pixel. The labels are assigned using kmeans (distance of
        each pixel's intensity to label means).
    """
    scale_min, scale_max = scale_range
    mus, labels, _ = kmeans(data.ravel(), k, scale_max=scale_max, scale_min=scale_min)
    labels = labels.reshape(data.shape)
    thetas = []
    for i in range(k):
        sigma = np.random.rand() * (scale_sigma - 1) + 1
        thetas.append([mus[i], sigma])

    return thetas, labels.astype(int)

def mrf_em(data, beta, k=3, max_iter=10^5, scale_range=(0, 100), scale_sigma=20,
            max_label_iter=100, njobs=1, map_labels=['csf', 'gray', 'white']):
    """
    Run MRF-EM.

    Input
    -----
    data : arrray (n, m)
        Image to segment.

    beta : float
        Parameter for determining how much two neighboring pixels should be
        forced to have the same label. Higher beta values correspond with more
        homogeneity.

    k : int, default=3
        Number of labels.

    max_iter : int , default=10^5
        Maximum number of iterations for EM.

    scale_range : tuple, default=(0, 100)
        (min, max) Range of values from which start kmeans.

    scale_sigma : float, default=20
        Sigma will be initialized with a random value between 1 and scale_sigma.

    max_label_iter : int, default=100
        Maximum number of iterations for ICM.

    njobs : int, default=1
        How many jobs to run in parallel when looking for the next set of
        most probable labels.

    Output
    ------
    thetas : list of lists
        Contains [mu, sigma] for each label.

    labels : int array (n, m)
        Labels for each pixel. The labels are assigned using kmeans (distance of
        each pixel's intensity to label means).
    """
    L = np.array(range(k))
    thetas, label = init_values(data, k, scale_range, scale_sigma)
    print('Initial thetas: ', thetas)

    # Get best labels for pixels
    for i in range(max_iter):
        print('Iteration', i)
        label = get_labels(data, thetas, label, L, beta, max_label_iter, njobs)
        thetas = update_thetas(data, thetas, beta, L, label)

    # Sort means by value for probability maps
    mus = [t[0] for t in thetas]
    sorted_idx = np.argsort(mus)

    # Find probability maps
    maps = {}
    norm = np.zeros(data.shape) # Norm for probabilities
    for ilabel, slabel in enumerate(map_labels):
        theta = thetas[sorted_idx[ilabel]]
        g = likelihood(data, theta)
        norm += g
        l_map = np.zeros(data.shape) # Output map for label
        # Find probability for every voxel
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                p_lx = p_label(ilabel, L, beta, [i, j], label)
                l_map[i, j] = g[i, j] * p_lx
        # Add to maps dictionary
        maps[slabel] = l_map
    # Normalize maps
    for slabel, m in maps.items():
        nl_map = m / norm
        nl_map[np.isnan(nl_map)] = 0 # For very low probabilities
        maps[slabel] = nl_map

    return thetas, label, maps
