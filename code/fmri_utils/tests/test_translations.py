""" py.test test for translations code

Run with:

    py.test motion_correction
"""
# import common modules
import numpy as np
import numpy.linalg as npl
from numpy.testing import assert_almost_equal
import scipy.ndimage as snd
from scipy.optimize import fmin_powell

# import translation functions to be tested
from fmri_utils.func_preproc.translations import optimize_trans_vol
# making a change to test github framework


def test_optimize_trans_vol():
    # Test optimization fo translation between two volumes

    # add an intentionally shifted volume by x, y, z

    # test to see whether to optimization of translation function
    #  indeed captures a very similar volume between the shifted and original

    return
