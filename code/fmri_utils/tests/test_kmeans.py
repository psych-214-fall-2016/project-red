""" py.test test for kmeans.py
Run with:
    py.test test_kmeans.py
"""

import numpy as np
from fmri_utils.segmentation.kmeans import *

def test_should_go():
    centers = [1, 2, 3, 4]
    centers_no_match = [1, 2, 2, 4]
    # Test iteration condition
    assert(should_go(centers, centers, 500, 499) == False)
    # Test all close
    assert(should_go(centers, centers, 499, 500) == False)
    assert(should_go(centers, centers_no_match, 499, 500) == True)

def test_get_labels():
    centers = [10, 20]
    x = np.concatenate((np.random.normal(10, 0.5, 10),
        np.random.normal(20, 0.5, 10)))
    labels = get_labels(x, centers)
    assert(np.allclose(labels[:10], np.zeros(10)))
    assert(np.allclose(labels[10:], np.ones(10)))

def test_get_centers():
    x = np.concatenate((np.ones(10)*10, np.ones(10)*20))
    labels = np.concatenate((np.zeros(10), np.ones(10)))
    centers = get_centers(x, labels, 2)
    assert(np.allclose(centers, [10, 20]))

def test_kmeans():
    x = np.concatenate((np.ones(10)*10, np.ones(10)*50))
    centers, _ = kmeans(x, k=2, scale=100)
    assert(len(centers) == 2)
    assert(sorted(centers) == [10, 20])
