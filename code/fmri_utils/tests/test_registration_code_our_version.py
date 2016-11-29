""" py.test test for registration/code_our_version.py

Run with:

    py.test test_code_our_version.py
"""

from os.path import dirname, join as pjoin

import nibabel as nib
import numpy as np

from fmri_utils.registration.shared import get_data_affine
from fmri_utils.registration.code_our_version import resample

MY_DIR = dirname(__file__)
TEMPLATE_FILENAME = 'mni_icbm152_t1_tal_nlin_asym_09a.nii'
ANAT_FILENAME = 'ds114_sub009_highres.nii'

def test_resample():
    #check subj anat resampled to template space
    template_path = pjoin(MY_DIR, TEMPLATE_FILENAME)
    anat_path = pjoin(MY_DIR, ANAT_FILENAME)

    static_data, static_affine = get_data_affine(template_path)
    moving_data, moving_affine = get_data_affine(anat_path)

    moving_new, moving_new_affine = resample(static_data, moving_data, static_affine, moving_affine)

    assert(np.array_equal(moving_new.shape, static_data.shape))
    assert(np.array_equal(moving_new_affine, static_affine))
