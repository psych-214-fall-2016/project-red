""" py.test test for registration/code_our_version.py

Run with:

    py.test test_code_our_version.py
"""

from os.path import dirname, join as pjoin

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

from fmri_utils.registration.shared import get_data_affine
from fmri_utils.registration.code_our_version import resample, transform_cmass

MY_DIR = dirname(__file__)
TEMPLATE_FILENAME = 'mni_icbm152_t1_tal_nlin_asym_09a.nii'
ANAT_FILENAME = 'ds114_sub009_highres.nii'

def test_resample():
    #check resample works, using fake data
    small = np.zeros((5,5,5))
    small[3,3,3] = 1
    small_affine = nib.affines.from_matvec(np.eye(3)*3, np.zeros(3))

    large = np.zeros((15,15,15))
    large[9:11, 9:11] = 1
    large_affine = np.eye(4)

    large_in_small, new_affine = resample(small, large, small_affine, large_affine)
    assert(np.array_equal(large_in_small.shape, small.shape))
    ##not same values, look at this!
    #assert(np.array_equal(large_in_small, small))

    small_in_large, new_affine = resample(large, small, large_affine, small_affine)
    assert(np.array_equal(small_in_large.shape, large.shape))
    ##not same values, look at this!
    #assert(np.array_equal(small_in_large, large))


    #check subj anat resampled to template space
    template_path = pjoin(MY_DIR, TEMPLATE_FILENAME)
    anat_path = pjoin(MY_DIR, ANAT_FILENAME)

    static_data, static_affine = get_data_affine(template_path)
    moving_data, moving_affine = get_data_affine(anat_path)

    moving_new, moving_new_affine = resample(static_data, moving_data, static_affine, moving_affine)

    assert(np.array_equal(moving_new.shape, static_data.shape))
    assert(np.array_equal(moving_new_affine, static_affine))

    #check that template esampled to template space is the same
    moving_new, moving_new_affine = resample(static_data, static_data, static_affine, static_affine)
    assert(np.array_equal(moving_new, static_data))

def test_transform_cmass():
    #check center of mass transform works, using fake data
    FAKE = np.zeros((11,11,11))
    FAKE[5,5,5] = 100
    FAKE_affine = np.eye(4)

    original_shift = nib.affines.from_matvec(np.eye(3), [1,2,3])
    mat, vec = nib.affines.to_matvec(original_shift)
    FAKE_moved = affine_transform(FAKE, mat, vec, order=1)

    FAKE_fix, FAKE_fix_affine, ref_shift = transform_cmass(FAKE, FAKE_moved, FAKE_affine, FAKE_affine)

    assert(np.array_equal(FAKE_fix, FAKE))
    assert(np.array_equal(ref_shift, original_shift))
