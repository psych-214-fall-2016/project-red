""" py.test test for mapping coordinates code

Run with:

    py.test fmri_utils/tests/test_coord_mapping.py
"""
# import common modules
import numpy as np
import numpy.linalg as npl
from numpy.testing import assert_almost_equal
import scipy.ndimage as snd
from scipy.optimize import fmin_powell
import nibabel as nib

from os.path import dirname, join as pjoin
MY_DIR = dirname(__file__)

EXAMPLE_FILENAME = 'ds114_sub009_t2r1.nii'

#from fmri_utils.func_preproc.optimize_rotations import optimize_rot_vol
#: Check import of rotations code
from fmri_utils.func_preproc.rotations import x_rotmat, y_rotmat, z_rotmat

from fmri_utils.func_preproc.optimize_map_coordinates import optimize_map_vol

def test_optimize_map_vol():
    # Test optimization of rotations between two volumes
    print(MY_DIR)
    example_path = pjoin(MY_DIR, EXAMPLE_FILENAME)
    #expected_values = np.loadtxt(pjoin(MY_DIR, 'global_signals.txt'))

    #img = nib.load('ds114_sub009_t2r1.nii')
    img = nib.load(example_path)
    data = img.get_data()
    vol0 = data[..., 4]
    vol1 = data[..., 5]

    #vol1_affine = img.affine

    # add an intentionally rotated volume by X, Y, Z
    X = x_rotmat(0.04)
    #- * radians around the y axis, then
    Y = y_rotmat(0.01)
    #- * radians around the z axis.
    Z = z_rotmat(-0.03)
    rotations = np.array([0.04, 0.01, -0.03])
    #- Mutiply matrices together to get matrix describing all 3 rotations
    M = Z.dot(Y.dot(X))

    translations = np.array([-1.,-0.7,0.4])
    test_params = np.append(translations, rotations)

    transformed_vol1 = snd.affine_transform(vol1, M, translations, order = 1)

    # test to see whether to optimization of resampling function
    #  indeed captures a very similar volume between the transformed and original

    #resampled_vol1, best_params = optimize_map_vol(vol1, transformed_vol1, img.affine)
    best_params = optimize_map_vol(vol1, transformed_vol1, img.affine)
    assert_almost_equal(best_params, test_params, decimal = 3)

    return
