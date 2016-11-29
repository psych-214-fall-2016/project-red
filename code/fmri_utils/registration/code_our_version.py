"""
code_our_version.py:

- resample: produce resampled image so same dims as another image

"""

import numpy as np
import numpy.linalg as npl
import nibabel as nib

from scipy.ndimage import affine_transform, measurements

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
    moving_in_stat : array shape (I, J, K)
        array with 3D from moving image resampled in static image space

    new_affine : array shape (4, 4)
        affine for new moving image (in static space) to ref

    """

    moving2static = npl.inv(moving_affine).dot(static_affine)
    mat, vec = nib.affines.to_matvec(moving2static)

    moving_in_stat = affine_transform(moving_data, mat, vec,
                                         output_shape=static_data.shape, order=1)

    new_affine = static_affine
    return moving_in_stat, new_affine


def transform_cmass(static_data, moving_data, static_affine, moving_affine):
    """ resample moving image in static image space
        --> matches center of mass of moving iamge to static image

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
    moving_in_stat : array shape (I, J, K)
        array with 3D from moving image file, resampled in static image space;
        has same affine as static image!

    new_affine : array shape (4, 4)
        affine for new moving image (in static space) to ref

    shift : array shape (4, 4)
        affine for changes in ref space
    """

    static_mat, static_vec = nib.affines.to_matvec(static_affine)
    moving_mat, moving_vec = nib.affines.to_matvec(moving_affine)

    static_cmass = np.array(measurements.center_of_mass(np.array(static_data)))
    moving_cmass = np.array(measurements.center_of_mass(np.array(moving_data)))

    static_cmass_in_ref = static_mat.dot(static_cmass) + static_vec
    moving_cmass_in_ref = moving_mat.dot(moving_cmass) + moving_vec

    diff_cmass_in_ref = static_cmass_in_ref - moving_cmass_in_ref

    shift = nib.affines.from_matvec(np.eye(3), diff_cmass_in_ref)
    updated_moving_affine = shift.dot(moving_affine)

    moving_in_stat, new_affine = resample(static_data, moving_data, static_affine, updated_moving_affine)

    return moving_in_stat, new_affine, shift
