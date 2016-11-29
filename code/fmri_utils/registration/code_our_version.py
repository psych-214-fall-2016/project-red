"""
code_our_version.py:

- resample: produce resampled image so same dims as another image

"""

import numpy as np
import numpy.linalg as npl
import nibabel as nib

from scipy.ndimage import affine_transform

from fmri_utils.registration.shared import get_data_affine


def resample(static_data, moving_data, static_affine, moving_affine):
    """ resample moving image in static image space

    Parameters
    ----------
    static_data : array shape (I, J, K)
        array with 3D data from static image file

     moving_data : array shape (I, J, K)
        array with 3D data from moving image file

    static_affine : array shape (4, 4)
        affine for static image file

    moving_affine : array shape (4, 4)
        affine for static image file

    Returns
    -------
    moving_in_stat_sp : array shape (I, J, K)
        array with 3D from moving image file, resampled in static image space;
        has same affine as static image!

    """

    moving2static = npl.inv(moving_affine).dot(static_affine)
    mat, vec = nib.affines.to_matvec(moving2static)

    moving_in_stat_sp = affine_transform(moving_data, mat, vec,
                                         output_shape=static_data.shape, order=1)
    return moving_in_stat_sp, static_affine
