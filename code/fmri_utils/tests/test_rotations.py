""" py.test test for rotations code

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

#from fmri_utils.func_preproc.translations import cost_at_xyz, optimize_trans_vol

# import translation functions to be tested
from fmri_utils.func_preproc.optimize_rotations import optimize_rot_vol
#: Check import of rotations code
from fmri_utils.func_preproc.rotations import x_rotmat, y_rotmat, z_rotmat

def test_optimize_rot_vol():
    # Test optimization of rotations between two volumes

    example_path = pjoin(MY_DIR, EXAMPLE_FILENAME)
    #expected_values = np.loadtxt(pjoin(MY_DIR, 'global_signals.txt'))

    import nibabel as nib
    #img = nib.load('ds114_sub009_t2r1.nii')
    img = nib.load(example_path)
    data = img.get_data()
    vol0 = data[..., 4]
    vol1 = data[..., 5]

    # add an intentionally rotated volume by X, Y, Z
    X = x_rotmat(0.03)
    #- * radians around the y axis, then
    Y = y_rotmat(0.09)
    #- * radians around the z axis.
    Z = z_rotmat(-0.05)
    #- Mutiply matrices together to get matrix describing all 3 rotations
    M = Z.dot(Y.dot(X))
    rotated_vol1 = snd.affine_transform(vol1, M)

    rotations = np.array([-0.03, -0.09, 0.05])

    # test to see whether to optimization of rotation function
    #  indeed captures a very similar volume between the rotated and original

    derotated_vol1, best_params = optimize_rot_vol(vol1, rotated_vol1)
    assert_almost_equal(best_params, rotations, decimal = 2)

    return
