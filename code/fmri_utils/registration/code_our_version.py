"""
code_our_version.py

"""
import os
from os.path import dirname, join as pjoin
import csv

import numpy as np
import numpy.linalg as npl
import nibabel as nib

from scipy.ndimage import affine_transform, measurements
from scipy.optimize import fmin_powell

from dipy.viz import regtools
import matplotlib.pyplot as plt

from fmri_utils.registration.shared import get_data_affine, make_rot_mat, decompose_rot_mat
from fmri_utils.func_preproc.rotations import x_rotmat, y_rotmat, z_rotmat

def resample(static_data, moving_data, static_affine, moving_affine):
    """
    resample moving image in static image space

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
    """
    get moving image affine, to use when resampling moving in static space
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
    """
    mutual information cost function: transforms moving image with updated
        values from optimizer and returns similartiy metric (negative
        mutual information)

    Parameters
    ----------
    parameters : vector length (N,)
        vector with variable parameters for optimizer

    subset : str
        'all' : all parameters are variable (N=3, 6, 9, or 15)
        'translations', 'rotations', 'scales', or 'shears' : only corresponding parametes are varible

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

    if subset=='all':
        temp_params = list(parameters)
    else:
        temp_params = fixed_parameters + list(parameters)

    #create affine from new params
    change_affine = params2affine(temp_params)
    updated_moving_affine = change_affine.dot(moving_affine)

    #resample with new affine
    moving_resampled = resample(static_data, moving_data, static_affine, updated_moving_affine)

    #get negative mutual information (static & new moving)
    neg_MI = neg_mutual_info(static_data, moving_resampled)

    return neg_MI



def transform_rigid(static_data, moving_data, static_affine, moving_affine, starting_affine, iterations, partial='all'):
    """
    get moving image affine, to use when resampling moving in static space
        --> does rigid (3 trans, 3 rot) alignment, max `iterations`

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

    iterations : int
        max number iterations in optimization

    partial : str
        'all' = best translations (3 params) -> best translations and rotations (6 params)
        'translations' = best translations only (3 params)
        'rotations' = best rotations only (3 params)

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
    if partial in ['translations']:
        best_translations = fmin_powell(MI_cost, params0[:3], args = (partial, [], static_data, moving_data, static_affine, moving_affine), maxiter = iterations)
        params1 = list(best_translations)
    elif partial in ['all','rotations']:
        params1 = params0

    params1 = params1 + rotations0

    # get best rotations
    if partial in ['all']:
        best_params = fmin_powell(MI_cost, params1, args = (partial, [], static_data, moving_data, static_affine, moving_affine), maxiter = iterations)
    elif partial in ['rotations']:
        best_rotations = fmin_powell(MI_cost, params1[3:], args = (partial, params1[:3], static_data, moving_data, static_affine, moving_affine), maxiter = iterations)
        best_params = params1[:3] + list(best_rotations)
    elif partial in ['translations']:
        best_params = params1

    # make best affine
    change_affine = params2affine(best_params)
    return change_affine

def transform_affine(static_data, moving_data, static_affine, moving_affine, starting_affine, iterations, partial='all'):
    """
    get moving image affine, to use when resampling moving in static space
        --> does affine (3 trans, 3 rot, 3 scale, 3 shear) alignment, max `iterations`

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

    iterations : int
        max number iterations in optimization

    partial : str
        'all' = best translations, rotations, and scales (9 params) -> best translations, rotations, scales, and shears (12 params)
        'translations' = best translations only (3 params)
        'rotations' = best rotations only (3 params)

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
    if partial in ['scales']:
        best_scales = fmin_powell(MI_cost, scales0, args = (partial, params0[:6], static_data, moving_data, static_affine, moving_affine), maxiter = iterations)
        params1 = params0[:6] + list(best_scales)
    elif partial in ['all', 'shears']:
        params1 = params0

    params1 = params1 + shears0

    # get best shears
    if partial in ['all']:
        best_params = list(fmin_powell(MI_cost, params1, args = (partial, [], static_data, moving_data, static_affine, moving_affine), maxiter = iterations))
    elif partial in ['scales']:
        best_params = params1
    elif partial in ['shears']:
        best_shears = fmin_powell(MI_cost, params1[9:], args = (partial, params1[:9], static_data, moving_data, static_affine, moving_affine), maxiter = iterations)
        best_params = params1[:9] + list(best_shears)

    #combine best params
    change_affine = params2affine(best_params)
    return change_affine


def mutual_info(static_data, moving_data, nbins):
    """
    get mutual information (MI) between 2 arrays
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


def neg_mutual_info(static, resampled):
    return (-1)*mutual_info(static, resampled, 32)

def params2affine(params):
    """
    create affine from list of parameters
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


def affine_registration(static_filename, moving_filename, SCALE, affines_dir, iterations):
    """
    does full affine registration (cmass -> translation -> rigid -> affine)
    takes static and moving files, downsampled by SCALE factor
    saves affines from each step as *.txt in output_affines dir

    Parameters
    ----------
    static_filename : str
        path to static file

    moving_filename : str
        path to moving file

    SCALE : float
        >0, rescale 3D array axes by SCALE factor (useful to downsample images for faster registration)

    affines_dir : str
        path to dir where affines are saved as *.txt

    iterations : int
        max number iterations in optimizations

    Returns
    -------
    final_affine : array shape (4, 4)
        best affine from full optimization

    """
    # extract affine file name from moving_filename
    affine_prefix = moving_filename.split('/')[-1]
    idx = affine_prefix.find('.nii')
    affine_prefix = affine_prefix[:idx]

    # load static and moving images, downsample
    static, static_affine = rescale_img(static_filename, SCALE)
    moving, moving_affine = rescale_img(moving_filename, SCALE)

    ## resample in static space
    init_affine = np.eye(4)
    save_affine(init_affine, affines_dir, affine_prefix+'_resampled.txt')

    ## do center of mass transform
    cmass_affine = transform_cmass(static, moving, static_affine, moving_affine)
    save_affine(cmass_affine, affines_dir, affine_prefix+'_cmass.txt')

    ## do translation transform
    translation_affine = transform_rigid(static, moving, static_affine, moving_affine, cmass_affine, iterations, 'translations')
    save_affine(translation_affine, affines_dir, affine_prefix+'_translation.txt')

    ## do rigid transform (translation & rotation)
    rigid_affine = transform_rigid(static, moving, static_affine, moving_affine, translation_affine, iterations, 'all')
    save_affine(rigid_affine, affines_dir, affine_prefix+'_rigid.txt')

    ## do full affine transform (translation, rotation, scaling & shearing)
    final_affine = transform_affine(static, moving, static_affine, moving_affine, rigid_affine, iterations, 'all')
    save_affine(final_affine, affines_dir, affine_prefix+'_sheared.txt')

    return final_affine


def rescale_img(img_filename, SCALE):
    """
    downsample image by SCALE

    Parameters
    ----------
    img_filename : str
        path to 3D image file

    SCALE : float
        >0, rescale 3D array axes by SCALE factor (useful to downsample images for faster registration)

    Returns
    -------
    img_scaled_data : array shape (I, J, K)
        array with downsampled 3D data

    img_scaled_affine : array shape (4, 4)
        affine for downsampled 3D array

    """
    # load data
    img_data, img_affine = get_data_affine(img_filename)

    # set downsample vars
    SCALE_affine = nib.affines.from_matvec(np.diagflat([1/SCALE]*3), np.zeros(3))
    img_scaled_shape = (np.array(img_data.shape)*SCALE).astype('int')
    img_scaled_affine = img_affine.dot(SCALE_affine)

    # resample
    img_scaled_data = resample(np.zeros(img_scaled_shape), img_data, img_scaled_affine, img_affine)

    return img_scaled_data, img_scaled_affine


def save_affine(affine, affines_dir, affine_filename):
    """
    save affine *.txt

    Parameters
    ----------
    affine : array shape (4, 4)
        affine for 3D array

    affines_dir : str
        dir where `affine_filename` will be saved

    affine_filename : str
        filename for output text file

    Returns
    -------
    None

    """
    # save affines
    f = open(pjoin(affines_dir, affine_filename), 'wt')
    f.write(str(affine))
    f.close()


def load_affine(affines_dir, affine_filename):
    """
    turn affine *.txt into numpy array

    Parameters
    ----------
    affines_dir : str
        dir where `affine_filename` is saved

    affine_filename : str
        filename for input text file

    Returns
    -------
    affine : array shape (4, 4)
        affine for 3D array

    """
    affine_txt = open(pjoin(affines_dir, affine_filename), 'rt')

    lines = [i.replace('[','').replace(']','').split() for i in affine_txt]
    lines_float = [[float(y) for y in row] for row in lines]

    affine = np.array(lines_float)
    return affine

def generate_transformed_images(static_filename, moving_filename, SCALE, affines_dir, output_dir, subset='both'):
    """
    applies affines_dir/*.txt affines to moving image
    saves transformed *.nii.gz and middle slice overlays *.png in output_dir

    **MUST USE SAME ARGS AS `affine_registration`

    Parameters
    ----------
    static_filename : str
        path to static file

    moving_filename : str
        path to moving file

    SCALE : float
        >0, rescale 3D array axes by SCALE factor (useful to downsample images for faster registration)

    affines_dir : str
        path to dir where affines are saved as *.txt

    output_dir : str
        path to dir where transformed *.nii.gz and *.png are saved

    subset : str
        'png' = only png overlays
        'nii' = only nii files
        'both' = png and nii files

    Returns
    -------
    None

    """

    # extract affine file name from moving_filename
    affine_prefix = moving_filename.split('/')[-1]
    idx = affine_prefix.find('.nii')
    affine_prefix = affine_prefix[:idx]

    # summary file, save info about how images are generate
    summary_file = open(pjoin(output_dir, affine_prefix + '_generated_images_summary.csv'), 'w')
    summary_wr = csv.writer(summary_file)
    summary_wr.writerow(['static_filename','moving_filename','transform_affine','negative_MI','generated_nii','generated_png'])

    # load static and moving images, downsample
    static, static_affine = rescale_img(static_filename, SCALE)
    moving, moving_affine = rescale_img(moving_filename, SCALE)

    affine_txt = [i for i in os.listdir(affines_dir) if i.find(affine_prefix)>-1 and i[-4:]=='.txt']

    for affine_filename in affine_txt:
        print('... applying ' + affine_filename)


        # get prefix for output files
        idx = affine_filename.find('.txt')
        prefix = affine_filename[:idx]

        # get affine values
        current_affine = load_affine(affines_dir, affine_filename)

        # resample moving image
        updated_moving_affine = current_affine.dot(moving_affine)
        resampled = resample(static, moving, static_affine, updated_moving_affine)

        # calculate neg_MI
        neg_MI = neg_mutual_info(static, resampled)

        # save resampled *.nii.gz images with static_affine
        nii_names = None
        if subset in ['nii', 'both']:
            nii_name = pjoin(output_dir, prefix+'.nii.gz')
            img = nib.Nifti1Image(resampled, static_affine)
            nib.save(img, nii_name)

        # save slice overlays with regtools from dipy.viz
        png_names = None
        if subset in ['png', 'both']:
            png_names = [None]*3
            for i in range(3):
                png_names[i] = pjoin(output_dir, prefix+'_'+str(i)+'.png')
                regtools.overlay_slices(static, resampled, None, i, 'Static', 'Moving', png_names[i])
            plt.close('all')

        record = [static_filename, moving_filename, affine_filename, neg_MI, nii_names, png_names]
        summary_wr.writerow(record)

    summary_file.close()


def main():
    # example usage: register sub-10159 anat to MNI template

    MY_DIR = dirname(__file__)
    reg_ex_dir = pjoin(MY_DIR,'..','..','..','data','registration_example_files')

    # register moving to static
    static_filename = pjoin(reg_ex_dir,'mni_icbm152_t1_tal_nlin_asym_09a_brain.nii.gz')
    moving_filename = pjoin(reg_ex_dir, 'sub-10159_T1w_brain.nii.gz')

    # set params, dirs
    SCALE = 0.25
    affines_dir = pjoin(reg_ex_dir, 'temp')
    iterations = 10
    output_dir = affines_dir

    # generate new affines
    affine_registration(static_filename, moving_filename, SCALE, affines_dir, iterations)

    # apply affines, generate images
    generate_transformed_images(static_filename, moving_filename, SCALE, affines_dir, output_dir)

    ''' note: can use saved .txt affines to generate images if they live
    in affines_dir; remember to use the same arguments for both functions
    for accurate transformations!!
    '''

if __name__ == '__main__':
    main()
