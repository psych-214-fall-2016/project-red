""" Test script for registration module: shared.py

Test with ``py.test test_registration_shared.py``.

"""

from os.path import dirname, join as pjoin

import nibabel as nib
import numpy as np

from fmri_utils.registration.shared import get_data_affine, make_rot_mat, decompose_rot_mat

MY_DIR = dirname(__file__)
EXAMPLE_FILENAME = 'ds107_sub012_t1r2_small.nii'

def test_get_data_affine():
    #check if works with sample data file
    example_path = pjoin(MY_DIR, EXAMPLE_FILENAME)
    data, affine = get_data_affine(example_path)

    img = nib.load(example_path)
    assert(np.array_equal(img.get_data(),data))
    assert(np.array_equal(img.get_affine(),affine))

def test_rotations():
    #check: no rotation
    test_rotations = [0,0,0]
    new_rot_mat = make_rot_mat(test_rotations)
    assert(np.array_equal(new_rot_mat, np.eye(3)))

    test_rot_mat = np.eye(3)
    new_rotations = decompose_rot_mat(test_rot_mat)
    assert(np.array_equal(new_rotations, [0,0,0]))

    #check: make mat, recover params
    test_rotations = [0.4, -0.3, 1]
    new_rot_mat = make_rot_mat(test_rotations)
    new_rotations = decompose_rot_mat(new_rot_mat)
    assert(np.array_equal(new_rotations, test_rotations))

    #check: recover params, make mat
    test_rot_mat = np.array([[ 0.    , -0.9239,  0.3827],[ 0.7071,  0.2706,  0.6533],[-0.7071,  0.2706,  0.6533]])
    new_rotations = decompose_rot_mat(test_rot_mat)
    new_rot_mat = make_rot_mat(new_rotations)
    assert(np.allclose(new_rot_mat, test_rot_mat, atol=1e7))
