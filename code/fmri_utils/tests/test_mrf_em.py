"""
Tests for mrf_em.py

@author: Christine Tseng
"""

import numpy as np
from fmri_utils.segmentation.mrf_em import log_likelihood, log_prior, get_pairs, log_posterior, should_go, get_best_label
from fmri_utils.segmentation.mrf_em import likelihood, get_neighbors, p_label, get_labels
from fmri_utils.segmentation.mrf_em import init_values, mrf_em


### TESTS FOR MRF FUNCTIONS ###

def test_log_likelihood():
    data = np.ones((2, 2))
    labels = np.array([[0, 0], [1, 1]])
    thetas = [[1, 1], [0, 1]]
    l = log_likelihood(thetas, data, labels)
    assert(l == 1)

def test_log_prior():
    pairs = [[0, 0], [1, 0]], [[0, 0], [0, 1]]
    labels = np.array([[0, 0], [1, 1]])
    beta = 0.5
    p = log_prior(pairs, labels, beta)
    assert(p == 0.5)

def test_get_pairs():
    data = np.ones((2, 2))
    pairs = get_pairs(data)
    real_pairs = [[[0, 0], [1, 0]], [[0, 0], [1, 1]], [[0, 0], [0, 1]],
                  [[0, 1], [1, 1]], [[0, 1], [1, 0]], [[1, 0], [1, 1]]]
    assert(real_pairs == pairs)

def test_log_posterior():
    data = np.ones((2, 2))
    thetas = [[1, 1], [0, 1]]
    labels = np.array([[0, 0], [1, 1]])
    beta = 0.5
    pairs = [[0, 0], [1, 0]], [[0, 0], [0, 1]]
    post = log_posterior(data, thetas, labels, beta, pairs)
    assert(post == 1.5)

def test_should_go():
    centers = [1, 2, 3, 4]
    centers_no_match = [1, 2, 2, 4]
    # Test iteration condition
    assert(should_go(centers, centers, 500, 499) == False)
    # Test all close
    assert(should_go(centers, centers, 499, 500) == False)
    assert(should_go(centers, centers_no_match, 499, 500) == True)

def test_get_best_labels():
    data = np.ones((2, 2))
    thetas = [[1, 0.1], [0, 0.1]]
    labels = np.array([[0, 0], [1, 1]])
    beta = 0.5
    pairs = [[0, 0], [1, 0]], [[0, 0], [0, 1]]
    p = 2
    L = [0, 1]
    l, min_energy = get_best_label(p, data, thetas, labels, L, beta, pairs)
    # Check correct label, check returns better labeling
    assert(l == 0)
    assert(min_energy <= log_posterior(data, thetas, labels, beta, pairs))

def test_get_labels():
    data = np.ones((2, 2))
    thetas = [[1, 0.1], [0, 0.1]]
    labels = np.array([[0, 0], [1, 1]])
    L = [0, 1]
    l = get_labels(data, thetas, labels, L, beta=0.5, max_iter=100,
                njobs=2)
    real_l = np.array([[0, 0], [0, 0]])
    assert(np.allclose(l, real_l))

### TESTS FOR EM FUNCTIONS ###

def test_likelihood():
    data = np.ones((2, 2))
    theta = [0, 1]
    l = likelihood(data, theta)
    real_l = (1 / np.sqrt(2 * np.pi)) * np.exp(np.ones((2, 2)) * -0.5)
    assert(np.allclose(l, real_l))

def test_get_neighbors():
    dshape = (3, 3)

    # Locations for testing
    loc1 = [0, 0] # Upper left corner edge case
    loc2 = [0, 2] # Upper right corner edge case
    loc3 = [2, 0] # Lower left corner edge case
    loc4 = [2, 2] # Lower right corner edge case
    loc5 = [1, 1] # Middle case

    # Neighbors
    loc1_n = [[1, 0], [0, 1], [1, 1]]
    loc2_n = [[1, 2], [0, 1], [1, 1]]
    loc3_n = [[1, 0], [2, 1], [1, 1]]
    loc4_n = [[1, 2], [2, 1], [1, 1]]
    loc5_n = [[0, 1], [2, 1], [1, 0], [1, 2], [0, 0], [0, 2], [2, 0], [2, 2]]

    assert(get_neighbors(loc1, dshape) == loc1_n)
    assert(get_neighbors(loc2, dshape) == loc2_n)
    assert(get_neighbors(loc3, dshape) == loc3_n)
    assert(get_neighbors(loc4, dshape) == loc4_n)
    assert(get_neighbors(loc5, dshape) == loc5_n)

def test_p_label():
    current_labels = np.array([[0, 0], [1, 1]])
    label = 1
    beta = 1
    loc = [0, 1]
    L = np.array([0, 1])
    p = p_label(label, L, beta, loc, current_labels)
    p_real = np.exp(-2) / (np.exp(-1) + np.exp(-2))
    assert(np.isclose(p, p_real))


### TEST MAIN FUNCTIONS ###

def test_init_values():
    data = np.concatenate((np.ones(10)*10, np.ones(10)*50))
    thetas, labels = init_values(data, k=2, scale_range=(0, 100), scale_sigma=10)
    mus = [t[0] for t in thetas]
    assert(len(labels) == 20)
    assert(sorted(mus) == [10, 50])

def test_mrf_em():
    data = np.vstack((np.ones(5)*[8, 9, 10, 11, 12],
                      np.ones(5)*[48, 49, 50, 51, 52]))
    thetas, labels, maps = mrf_em(data, beta=0.5, k=2, max_iter=10^5, scale_range=(0, 100),
                        scale_sigma=10, max_label_iter=100, njobs=1,
                        map_labels=['one', 'two'])
    mus = [t[0] for t in thetas]
    # Two possbilities for labels
    l1 = np.vstack((np.zeros(5), np.ones(5)))
    l2 = np.vstack((np.ones(5), np.zeros(5)))
    # Check means, labels are correct
    assert(np.allclose(sorted(mus), [10, 50], atol=0.1))
    assert(np.allclose(labels, l1) or np.allclose(labels, l2))
    assert(len(maps) == 2)
