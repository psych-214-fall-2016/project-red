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
    change_affine : array shape (4, 4)
        new affine to adjust moving; updated_moving_affine = change_affine.dot(moving_affine)

    """

    static_mat, static_vec = nib.affines.to_matvec(static_affine)
    moving_mat, moving_vec = nib.affines.to_matvec(moving_affine)

    static_cmass = np.array(measurements.center_of_mass(np.array(static_data)))
    moving_cmass = np.array(measurements.center_of_mass(np.array(moving_data)))

    static_cmass_in_ref = static_mat.dot(static_cmass) + static_vec
    moving_cmass_in_ref = moving_mat.dot(moving_cmass) + moving_vec

    diff_cmass_in_ref = static_cmass_in_ref - moving_cmass_in_ref

    change_affine = nib.affines.from_matvec(np.eye(3), diff_cmass_in_ref)

    return change_affine

def MI_cost(parameters, subset, fixed_parameters, static_data, moving_data, static_affine, moving_affine):
    """ mutual information cost function: transforms moving image with updated
        values from optimizer and returns similartiy metric (negative
        mutual information)

    Parameters
    ----------
    parameters : vector length (N,)
        vector with variable parameters for optimizer

    subset : str
        "all" : all parameters are variable (N=3, 6, 9, or 15)
        "translations", "rotations", "scales", or "shears" : only corresponding parametes are varible

    fixed_parameters : list length(N0,)
        vector with fixed parameters for optimizer
        temp_params = fixed_parameters + list(parameters)
        -> temp_params = [3 translations] + [3 rotations] + [3 scales] + [6 shears]
        -> len(temp_params) in [3,6,9,15]; includes increasingly more params

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
    neg_MI : float
        negative mutual information value

    """

    if subset=="all":
        temp_params = list(parameters)
    else:
        temp_params = fixed_parameters + list(parameters)

    #create affine from new params
    change_affine = params2affine(temp_params)
    updated_moving_affine = change_affine.dot(moving_affine)

    #resample with new affine
    moving_resampled = resample(static_data, moving_data, static_affine, updated_moving_affine)

    #get negative mutual information (static & new moving)
    neg_MI = (-1)*mutual_info(static_data, moving_resampled, 32)

    return neg_MI

def transform_rigid(static_data, moving_data, static_affine, moving_affine, starting_affine, iter, partial="all"):
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

    partial : str
        "all" = best translations (3 params) -> best translations and rotations (6 params)
        "translations" = best translations only (3 params)
        "rotations" = best rotations only (3 params)

    Returns
    -------
    change_affine : array shape (4, 4)
        new affine to adjust moving; updated_moving_affine = change_affine.dot(moving_affine)

    """
    # get starting guess
    mat0, vec0 = nib.affines.to_matvec(starting_affine)
    rotations0 = list(decompose_rot_mat(mat0))
    translations0 = list(vec0)

    params0 = translations0

    # get best translations
    if partial in ["translations"]:
        best_translations = fmin_powell(MI_cost, params0[:3], args = (partial, [], static_data, moving_data, static_affine, moving_affine), maxiter = iter)
        params1 = list(best_translations)
    elif partial in ["all","rotations"]:
        params1 = params0

    params1 = params1 + rotations0

    # get best rotations
    if partial in ["all"]:
        best_params = fmin_powell(MI_cost, params1, args = (partial, [], static_data, moving_data, static_affine, moving_affine), maxiter = iter)
    elif partial in ["rotations"]:
        best_rotations = fmin_powell(MI_cost, params1[3:], args = (partial, params1[:3], static_data, moving_data, static_affine, moving_affine), maxiter = iter)
        best_params = params1[:3] + list(best_rotations)
    elif partial in ["translations"]:
        best_params = params1

    # make best affine
    change_affine = params2affine(best_params)
    return change_affine

def transform_affine(static_data, moving_data, static_affine, moving_affine, starting_affine, iter, partial="all"):
    """ get moving image affine, to use when resampling moving in static space
        --> does affine (3 trans, 3 rot, 3 scale, 3 shear) alignment, max "iter" iterations

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

    partial : str
        "all" = best translations, rotations, and scales (9 params) -> best translations, rotations, scales, and shears (12 params)
        "translations" = best translations only (3 params)
        "rotations" = best rotations only (3 params)

    Returns
    -------
    change_affine : array shape (4, 4)
        new affine to adjust moving; updated_moving_affine = change_affine.dot(moving_affine)

    """

    # get starting guess
    mat0, vec0 = nib.affines.to_matvec(starting_affine)
    rotations0 = list(decompose_rot_mat(mat0))
    translations0 = list(vec0)
    scales0 = [1]*3
    shears0 = [0]*3

    params0 = translations0 + rotations0 + scales0

    # get best scales
    if partial in ["scales"]:
        best_scales = fmin_powell(MI_cost, scales0, args = (partial, params0[:6], static_data, moving_data, static_affine, moving_affine), maxiter = iter)
        params1 = params0[:6] + list(best_scales)
    elif partial in ["all", "shears"]:
        params1 = params0

    params1 = params1 + shears0

    # get best shears
    if partial in ["all"]:
        best_params = list(fmin_powell(MI_cost, params1, args = (partial, [], static_data, moving_data, static_affine, moving_affine), maxiter = iter))
    elif partial in ["scales"]:
        best_params = params1
    elif partial in ["shears"]:
        best_shears = fmin_powell(MI_cost, params1[9:], args = (partial, params1[:9], static_data, moving_data, static_affine, moving_affine), maxiter = iter)
        best_params = params1[:9] + list(best_shears)

    #combine best params
    change_affine = params2affine(best_params)
    return change_affine


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



def params2affine(params):
    """ create affine from list of parameters
    Parameters
    ----------
    params : list length N
        list of parameters; default no change if parameters not given
        N=3 -> [3 translations] (+ [0]*3 + [1]*3 + [0]*3)
        N=6 -> [3 translations] + [3 rotations] (+ [1]*3 + [0]*3)
        N=9 -> [3 translations] + [3 rotations] + [3 scales] (+ [0]*3)
        N=15 -> [3 translations] + [3 rotations] + [3 scales] + [3 shears]

    Returns
    -------
    affine : array shape (4, 4)
        affine from parameters

    """
    #translation vector
    translations = params[:3]

    #rotation matrix
    if len(params)>3:
        rotations = params[3:6]
    else:
        rotations = [0]*3
    rot_mat = make_rot_mat(rotations)

    #scaling matrix
    if len(params)>6:
        scales = params[6:9]
    else:
        scales = [1]*3
    scale_mat = np.diagflat(scales)

    #shearing matrix
    if len(params)>9:
        shears = params[9:]
    else:
        shears = [0]*3
    shear_mat = np.eye(3)
    shear_mat[0,1] = shears[0]
    shear_mat[0,2] = shears[1]
    shear_mat[1,2] = shears[2]

    updated_rot_mat = rot_mat.dot(scale_mat).dot(shear_mat)
    affine = nib.affines.from_matvec(updated_rot_mat,translations)

    return affine




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
    nlevels = len(level_iters)

    static_shape = np.array(static_data.shape)

    for i in range(nlevels):
        print('working on level 1')
        static_resize_empty = np.zeros(static_shape/factors[i])
        static_resize_affine = nib.affine.from_matvec(np.eye(3)*factors[i], [0,0,0])

        static_resize_data = resample(static_resize_empty, static_data, np.eye(4), static_resize_affine)
