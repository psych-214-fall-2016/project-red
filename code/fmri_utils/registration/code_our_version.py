"""
code_our_version.py:

- resample: produce resampled image so same dims as another image

"""

import numpy as np
import numpy.linalg as npl
import nibabel as nib

from scipy.ndimage import affine_transform, measurements
from scipy.optimize import fmin_powell

from fmri_utils.registration.shared import get_data_affine
from fmri_utils.func_preproc.rotations import x_rotmat, y_rotmat, z_rotmat

def resample(static_data, moving_data, static_affine, moving_affine):
    """ resample moving image in static image space

    Parameters
    ----------
    static_data : array shape (I, J, K)
        array with 3D data from static image

     moving_data : array shape (I, J, K)
        array with 3D data from moving image

    static_affine : array shape (4, 4)
        affine for static image

    moving_affine : array shape (4, 4)
        affine for moving image

    Returns
    -------
    moving_in_stat : array shape (I, J, K)
        array with 3D from moving image resampled in static image space

    new_affine : array shape (4, 4)
        affine for new moving image (in static space) to ref

    """

    moving2static = npl.inv(moving_affine).dot(static_affine)
    mat, vec = nib.affines.to_matvec(moving2static)

    moving_in_stat = affine_transform(moving_data, mat, vec, output_shape=static_data.shape, order=1)

    new_affine = static_affine
    return moving_in_stat, new_affine


def transform_cmass(static_data, moving_data, static_affine, moving_affine):
    """ get moving image affine, to use when resampling moving in static space
        --> matches center of mass of moving image to static image (in ref space)

    Parameters
    ----------
    static_data : array shape (I, J, K)
        array with 3D data from static image

    moving_data : array shape (I, J, K)
        array with 3D data from moving image

    static_affine : array shape (4, 4)
        affine for static image

    moving_affine : array shape (4, 4)
        starting affine for mvoing image

    Returns
    -------
    updated_moving_affine : array shape (4, 4)
        new affine for moving image to ref

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

    return updated_moving_affine


def transform_rigid(static_data, moving_data, static_affine, moving_affine, iter):
    """ get moving image affine, to use when resampling moving in static space
        --> does rigid (3 trans, 3 rot) alignment, max "iter" iterations

    Parameters
    ----------
    static_data : array shape (I, J, K)
        array with 3D data from static image

    moving_data : array shape (I, J, K)
        array with 3D data from moving image

    static_affine : array shape (4, 4)
        affine for static image

    moving_affine : array shape (4, 4)
        starting affine for static moving

    iter : int
        max number iterations in optimization

    Returns
    -------
    updated_moving_affine : array shape (4, 4)
        new affine for moving image to ref

    """

    def MI_cost_rotation(rotations):
        #get MI for rotated moving_data
        moving_rotated = apply_rotation(moving_data, rotations)
        return mutual_info(static_data, moving_rotated, static_affine, moving_affine, 32)

    def MI_cost_translation(translations):
        #get MI for translated moving_data
        moving_translated = apply_translation(moving_data, rotations)
        return mutual_info(static_data, moving_translated, static_affine, moving_affine, 32)

    best_rotations = fmin_powell(MI_cost_rotation, [0,0,0], maxiter = iter)

    best_translations = fmin_powell(MI_cost_translation, [0,0,0], maxiter = iter)

    r_x, r_y, r_z = best_rotations
    best_rotations_matrix = z_rotmat(r_z).dot(y_rotmat(r_y)).dot(x_rotmat(r_x))

    updated_moving_affine = nib.affines.from_matvec(best_rotations_matrix, best_translations)

    return updated_moving_affine


def mutual_info(static_data, moving_data, static_affine, moving_affine, nbins):
    """ get mutual information (MI) between 2 arrays
    Parameters
    ----------
    img1 : array shape (I, J, ...)
        array of image 1

    img2 : array shape (I, J, ...)
        array of image 2

    nbins : int
        number bins for MI

    Returns
    -------
    MI : float
        mutual information value

    """
    moving_resampled, new_affine = resample(static_data, moving_data, static_affine, moving_affine)

    hist_2d, x_edges, y_edges = np.histogram2d(static_data.ravel(), moving_resampled.ravel(), bins=nbins) #get bin counts

    hist_2d_p = hist_2d/float(hist_2d.sum()) #p(x,y)
    nzs = hist_2d_p > 0 #idx for cells>0

    px = hist_2d_p.sum(axis=1) #marginal over y
    py = hist_2d_p.sum(axis=0) #marginal over x

    px_py = px[:,None] * py[None,:] #p(x)*p(y)
    MI = (hist_2d_p[nzs] * np.log(hist_2d_p[nzs]/px_py[nzs])).sum()

    return MI

def apply_translation(img, translations):
        """ apply translation to image

        Parameters
        ----------
        img : array shape (I, J, K)
            array with 3D data from original image

        translations : vector shape (3,)
            t_x, t_y, t_z voxels on axes

        Returns
        -------
        img_translated : array shape (I, J, K)
            array with 3D data of translated image

        """

        img_translated = affine_transform(img, np.eye(3), translations, order=1)
        return img_translated


def apply_rotation(img, rotations):
    """ apply rotation to image

    Parameters
    ----------
    img : array shape (I, J, K)
        array with 3D data from original image

    rotations : vector shape (3,)
        r_x, r_y, r_z radians around axes

    Returns
    -------
    img_rotated : array shape (I, J, K)
        array with 3D data of rotated image

    """
    r_x, r_y, r_z = rotations
    rotations_mat = z_rotmat(r_z).dot(y_rotmat(r_y)).dot(x_rotmat(r_x))

    img_rotated = affine_transform(img, rotations_mat, np.zeros(3), order=1)
    return img_rotated
