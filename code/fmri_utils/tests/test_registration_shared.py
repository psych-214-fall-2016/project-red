""" Test script for registration module: shared.py

Test with ``py.test test_registration_shared.py``.

"""

from os.path import dirname, join as pjoin

MY_DIR = dirname(__file__)
EXAMPLE_FILENAME = 'ds107_sub012_t1r2_small.nii'

import nibabel as nib
import numpy as np

from fmri_utils.registration.shared import get_data_affine

def test_get_data_affine():
    #check if works with sample data file
    example_path = pjoin(MY_DIR, EXAMPLE_FILENAME)
    data, affine = get_data_affine(example_path)

    img = nib.load(example_path)
    assert(np.array_equal(img.get_data(),data))
    assert(np.array_equal(img.get_affine(),affine))
