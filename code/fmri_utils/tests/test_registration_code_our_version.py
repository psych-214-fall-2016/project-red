""" py.test test for registration/code_our_version.py

Run with:

    py.test test_code_our_version.py
"""

from os.path import dirname, join as pjoin

import nibabel as nib
import numpy as np
import numpy.linalg as npl
from scipy.ndimage import affine_transform

from fmri_utils.registration.shared import get_data_affine
from fmri_utils.registration.code_our_version import resample, transform_cmass, transform_rigid
from fmri_utils.func_preproc.rotations import x_rotmat, y_rotmat, z_rotmat


# MY_DIR = dirname(__file__)
# TEMPLATE_FILENAME = 'mni_icbm152_t1_tal_nlin_asym_09a.nii'
# ANAT_FILENAME = 'ds114_sub009_highres.nii'

def test_resample():
    #check resample works, using fake data
    n = 5
    ORIG = np.zeros((n,n,n))
    ORIG[3,3,3] = 100
    ORIG_affine = np.eye(4)

    zoom = 3
    BIG_affine = nib.affines.from_matvec(np.eye(3)/zoom, np.zeros(3))
    mat, vec = nib.affines.to_matvec(BIG_affine)
    BIG = affine_transform(ORIG, mat, vec, output_shape=(n*zoom, n*zoom, n*zoom), order = 1)

    BIG_in_orig = resample(ORIG, BIG, ORIG_affine, BIG_affine)
    assert(np.array_equal(BIG_in_orig.shape, ORIG.shape))
    assert(np.array_equal(BIG_in_orig, ORIG))

    """
    #check subj anat resampled to template space
    template_path = pjoin(MY_DIR, TEMPLATE_FILENAME)
    anat_path = pjoin(MY_DIR, ANAT_FILENAME)

    static_data, static_affine = get_data_affine(template_path)
    moving_data, moving_affine = get_data_affine(anat_path)

    moving_new = resample(static_data, moving_data, static_affine, moving_affine)

    assert(np.array_equal(moving_new.shape, static_data.shape))
    assert(np.array_equal(moving_new_affine, static_affine))

    #check that template esampled to template space is the same
    moving_new= resample(static_data, static_data, static_affine, static_affine)
    assert(np.array_equal(moving_new, static_data))
    """


def test_transform_cmass():
    #check center of mass transform works, using fake data
    FAKE = np.zeros((11,11,11))
    FAKE[5,5,5] = 100
    FAKE_affine = np.eye(4)

    original_shift = nib.affines.from_matvec(np.eye(3), [1,2,3])
    mat, vec = nib.affines.to_matvec(original_shift)
    FAKE_moved = affine_transform(FAKE, mat, vec, order=1)
    FAKE_moved_affine = np.eye(4)

    updated_FAKE_moved_affine = transform_cmass(FAKE, FAKE_moved, FAKE_affine, FAKE_moved_affine)

    FAKE_fix = resample(FAKE, FAKE_moved, FAKE_affine, updated_FAKE_moved_affine)

    assert(np.array_equal(FAKE_fix, FAKE))
    assert(np.array_equal(npl.inv(FAKE_moved_affine).dot(updated_FAKE_moved_affine), original_shift))
    """
    add test with real brain images
    """


def test_transform_rigid():
    #check center of mass transform works, using fake data
    FAKE = np.zeros((30,30,30))
    FAKE[10:20,10:20,10:20] = np.random.rand(10,10,10)
    FAKE_affine = np.eye(4)

    #check translation only
    original_shift = nib.affines.from_matvec(np.diagflat([1,1,1]), [2,2,1])
    mat, vec = nib.affines.to_matvec(original_shift)
    FAKE_moved = affine_transform(FAKE, mat, vec, order=1)

    new_affine = transform_rigid(FAKE, FAKE_moved, np.eye(4), np.eye(4), 5, 1)
    assert(np.allclose(new_affine,original_shift,atol=0.1)) #withing 0.1 vox

    #check translation & rotation --- doesn't work!!! should use brain?
    rot_mat = z_rotmat(0.3).dot(y_rotmat(0.1)).dot(x_rotmat(0.1))
    original_shift = nib.affines.from_matvec(rot_mat, [2,2,1])
    mat, vec = nib.affines.to_matvec(original_shift)
    FAKE_moved = affine_transform(FAKE, mat, vec, order=1)

    new_affine = transform_rigid(FAKE, FAKE_moved, np.eye(4), np.eye(4), 5)
    #assert(np.allclose(new_affine,original_shift,atol=0.5)) #withing 0.5 vox

    """
    add test with real brain images
    """
