"""
code_our_version.py

"""

import numpy as np
import numpy.linalg as npl
import nibabel as nib

from scipy.ndimage import affine_transform, measurements
from scipy.optimize import fmin_powell

from fmri_utils.registration.shared import get_data_affine, make_rot_mat, decompose_rot_mat
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


    """

    moving2static = npl.inv(moving_affine).dot(static_affine)
    mat, vec = nib.affines.to_matvec(moving2static)

    moving_in_stat = affine_transform(moving_data, mat, vec, output_shape=static_data.shape, order=1)

    return moving_in_stat


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


def transform_rigid(static_data, moving_data, static_affine, moving_affine, starting_affine, iter, partial=0):
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

    starting_affine : array shape (4, 4)
        first guess for affine

    iter : int
        max number iterations in optimization

    partial : int, flag
        0 = best translations + rotations
        1 = best translations

    Returns
    -------
    updated_moving_affine : array shape (4, 4)
        new affine for moving image to ref

    """

    def MI_cost_translation(translations):
        ## cost function for translations using MI
        #create affine from new params
        shift_affine = nib.affines.from_matvec(np.eye(3), translations)
        updated_moving_affine = moving_affine.dot(shift_affine)

        #resample with new affine
        moving_resampled = resample(static_data, moving_data, static_affine, updated_moving_affine)

        #get negative mutual information (static & new moving)
        neg_MI = (-1)*mutual_info(static_data, moving_resampled, 64)

        return neg_MI

    def MI_cost_add_rotation(parameters):
        ## cost function for rotations using MI
        translations = parameters[:3]
        rotations = parameters[3:]

        #create affine from new params
        rot_mat = make_rot_mat(rotations)
        shift_affine = nib.affines.from_matvec(rot_mat,translations)

        updated_moving_affine = moving_affine.dot(shift_affine)

        #resample with new affine
        moving_resampled = resample(static_data, moving_data, static_affine, updated_moving_affine)

        #get negative mutual information (static & new moving)
        neg_MI = (-1)*mutual_info(static_data, moving_resampled, 32)

        return neg_MI

    # get starting guess
    mat0, vec0 = nib.affines.to_matvec(starting_affine)
    rotations0 = decompose_rot_mat(mat0)
    translations0 = list(vec0)

    # get best translations
    best_translations = fmin_powell(MI_cost_translation, translations0, maxiter = iter)

    #get best translations + rotations
    init_params = list(best_translations) + rotations0
    if partial==0:
        best_params = fmin_powell(MI_cost_add_rotation, init_params, maxiter = iter)
    else:
        best_params = init_params

    # combine best translations and rotations
    best_rotations_mat = make_rot_mat(best_params[3:])
    updated_moving_affine = nib.affines.from_matvec(best_rotations_mat, best_params[:3])

    return updated_moving_affine

def transform_affine(static_data, moving_data, static_affine, moving_affine, starting_affine, iter, partial=0):
    """ get moving image affine, to use when resampling moving in static space
        --> does affine (3 trans, 3 rot, 3 scale, 6 shear) alignment, max "iter" iterations

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

    starting_affine : array shape (4, 4)
        first guess for affine

    iter : int
        max number iterations in optimization

    partial : int, flag
        0 = best translations + rotations + scales + shears
        1 = best translations + rotations + scales

    Returns
    -------
    updated_moving_affine : array shape (4, 4)
        new affine for moving image to ref

    """


    def MI_cost_add_scales(parameters):
        ## cost function for rotations using MI
        translations = parameters[:3]
        rotations = parameters[3:6]
        scales = parameters[6:]

        #create affine from new params
        rot_mat = make_rot_mat(rotations)
        scale_mat = np.diagflat(scales)

        updated_rot_mat = rot_mat.dot(scale_mat)
        shift_affine = nib.affines.from_matvec(update_rot_mat,translations)
        updated_moving_affine = moving_affine.dot(shift_affine)

        #resample with new affine
        moving_resampled = resample(static_data, moving_data, static_affine, updated_moving_affine)

        #get negative mutual information (static & new moving)
        neg_MI = (-1)*mutual_info(static_data, moving_resampled, 32)

        return neg_MI

    def MI_cost_add_shears(parameters):
        ## cost function for rotations using MI
        translations = parameters[:3]
        rotations = parameters[3:6]
        scales = parameters[6:9]
        shears = parameters[9:]

        #create affine from new params
        rot_mat = make_rot_mat(rotations)
        scale_mat = np.diagflat(scales)
        shear_mat = make_shear_mat(shears)

        updated_rot_mat = rot_mat.dot(scale_mat).dot(shear_mat)
        shift_affine = nib.affines.from_matvec(update_rot_mat,translations)
        updated_moving_affine = moving_affine.dot(shift_affine)

        #resample with new affine
        moving_resampled = resample(static_data, moving_data, static_affine, updated_moving_affine)

        #get negative mutual information (static & new moving)
        neg_MI = (-1)*mutual_info(static_data, moving_resampled, 32)

        return neg_MI


    # get starting guess
    mat0, vec0 = nib.affines.to_matvec(starting_affine)
    rotations0 = decompose_rot_mat(mat0)
    translations0 = list(vec0)

    # get best translations + rotations + scales
    init_params = list(vec0) + list(rotations0) + [0,0,0]
    best_params = fmin_powell(MI_cost_add_scales, init_params, maxiter = iter)

    #get best translations + rotations + scales + shears
    init_params_shear = list(best_params) + [0,0,0,0,0,0]
    best_params = fmin_powell(MI_cost_add_shears, init_params, maxiter = iter)

    #combine best params
    translations = best_parameters[:3]
    rotations = best_parameters[3:6]
    scales = best_parameters[6:9]
    shears = best_parameters[9:]

    rot_mat = make_rot_mat(rotations)
    scale_mat = np.diagflat(scales)
    shear_mat = make_shear_mat(shears)

    updated_rot_mat = rot_mat.dot(scale_mat).dot(shear_mat)
    updated_moving_affine = nib.affines.from_matvec(update_rot_mat,translations)

    return updated_moving_affine
    

def mutual_info(static_data, moving_data, nbins):
    """ get mutual information (MI) between 2 arrays
    Parameters
    ----------
    static_data : array shape (I, J, ...)
        array of image 1

    moving_data : array shape (I, J, ...)
        array of image 2

    nbins : int
        number bins for MI

    Returns
    -------
    MI : float
        mutual information value

    """

    hist_2d, x_edges, y_edges = np.histogram2d(static_data.ravel(), moving_data.ravel(), bins=nbins) #get bin counts

    hist_2d_p = hist_2d/float(hist_2d.sum()) #p(x,y)
    nzs = hist_2d_p > 0 #idx for cells>0

    px = hist_2d_p.sum(axis=1) #marginal over y
    py = hist_2d_p.sum(axis=0) #marginal over x

    px_py = px[:,None] * py[None,:] #p(x)*p(y)
    MI = (hist_2d_p[nzs] * np.log(hist_2d_p[nzs]/px_py[nzs])).sum()

    return MI

def make_shear_mat(shears):
    ## make shear matrix from 6 values
    shear_mat = np.zeros((3,3))
    shear_mat[0,1] = shears[0]
    shear_mat[0,2] = shears[1]
    shear_mat[1,0] = shears[2]
    shear_mat[1,2] = shears[3]
    shear_mat[2,0] = shears[4]
    shear_mat[2,1] = shears[5]

    return shear_mat


'''
def pyramid(static_data, moving_data, static_affine, moving_affine, transformation, level_iters, sigmas, factors):
    """ apply transformation optimization using multiple levels (gaussian pyramid)
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

    transformation : function
        transformation function (e.g. transform_rigid, transform_afffine)

    level_iters : list of ints, length N
        max number of iterations at each level of the pyramid

    sigmas : list of ints, length N
        sigma for spatial smoothing at each level of the pyramid

    factors : list of ints, length N
        voxel factor at each level of pyramid (e.g. [4, 2] -> #/4, #/2 voxel dimensions)

    Returns
    -------
    updated_moving_affine : array shape (4, 4)
        new affine for moving image to ref

    """

    assert(len(level_iters)==len(sigmas) & len(level_iters)==len(factors))
'''
