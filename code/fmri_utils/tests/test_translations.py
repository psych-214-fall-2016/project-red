""" py.test test for translations code

Run with:

    py.test fmri_utils
"""
# import common modules
import numpy as np
import numpy.linalg as npl
from numpy.testing import assert_almost_equal
import scipy.ndimage as snd
from scipy.optimize import fmin_powell

from os.path import dirname, join as pjoin
MY_DIR = dirname(__file__)
EXAMPLE_FILENAME = 'ds114_sub009_t2r1.nii'

# import translation functions to be tested
from fmri_utils.func_preproc.translations import cost_at_xyz, optimize_trans_vol
# making a change to test github framework


def test_optimize_trans_vol():
    # Test optimization of translation between two volumes

    example_path = pjoin(MY_DIR, EXAMPLE_FILENAME)
    #expected_values = np.loadtxt(pjoin(MY_DIR, 'global_signals.txt'))

    import nibabel as nib
    #img = nib.load('ds114_sub009_t2r1.nii')
    img = nib.load(example_path)
    data = img.get_data()
    # add an intentionally shifted volume by x, y, z
    vol0 = data[..., 4]
    vol1 = data[..., 5]
    #shift vol1 by parameters, test whether code properly detects and returns
    # these translation parameters
    shifted_vol1 = np.zeros(vol1.shape)
    shifted_vol1[1:, 3:, 2:] = vol1[:-1, :-3, :-2]
    translations = np.array([-1., -3., -2.])

    # test to see whether to optimization of translation function
    #  indeed captures a very similar volume between the shifted and original

    unshifted_vol1, best_params = optimize_trans_vol(vol0, shifted_vol1)
    assert_almost_equal(best_params, translations, decimal = 2)

    return
