""" Volume realignment script

Input:
- Raw EPI time course (4D NifTi file)

Output:
- Rigid body motion parameters
  - x, y, z translations
  - X, Y, Z rotations (pitch, yaw, roll)
  - plot of motion parameters across volume
- Mean functional volume (3D NifTi file)"""

# - compatibility with Python 2
from __future__ import print_function  # print('me') instead of print 'me'
from __future__ import division  # 1/2 == 0.5, not 0
# - import common modules
import numpy as np  # the Python array package
import matplotlib.pyplot as plt  # the Python plotting package
import scipy.stats as stats
import scipy.ndimage as snd
import nibabel as nib
import nibabel.processing as proc
import os.path
from os.path import dirname, join as pjoin

# import translations and rotation optimization functions
from fmri_utils.func_preproc.translations import optimize_trans_vol
from fmri_utils.func_preproc.optimize_rotations import optimize_rot_vol, apply_rotations
from fmri_utils.func_preproc.optimize_map_coordinates import optimize_map_vol




def volume_4D_realign(img_path, reference = 0, smooth_fwhm = 0, prefix = 'r', drop_vols = 0):
    """
    Input:
    - 4D .nii img file
    - ref_vol (0 for fist volume, 1 for middle), default first
    - fwhm smoothing (value in mm, default is 0)
    - prefix of new img file to be saved
    - drop any volumes? index of volume, before which all volumes will be dropped (default = 0)

    Output:
    - 4D .nii img file with realigned data (and prefix)
    - 2d array (volume x 6) of motion parameters
    """
    img_dir, img_name = os.path.split(img_path)

    img = nib.load(img_path)

    # smooth image, if designated
    if smooth_fwhm > 0:
        fwhm = smooth_fwhm
        img = proc.smooth_image(img, 5)

    data = img.get_data()

    # Dropping volumes, specified by input of drop_vols
    #data = data[...,drop_vols:]
    #data = data[...,:10] # for testing purposes
    n_vols = data.shape[-1]


    print('Number of volumes to realign:', n_vols)
    # set whether the reference should be the first volume or middle volume
    if reference == 0:
        ref_vol = data[...,0]
    elif reference == 1:
        mid_index = int(n_vols/2)
        mid_vol = data[...,mid_index]
        ref_vol = mid_vol

    # array to which the realignment parameters for each of the 6 rigid body parameters
    #  will be added for each volume
    realign_params = np.zeros((n_vols, 6))
    realigned_data = np.zeros((data.shape))
    for i in range(n_vols):
        # use translation and rotation functions for each volume to the reference
        #translated_vol, trans_params = optimize_trans_vol(ref_vol, data[...,i])
        #rotated_vol, rot_params = optimize_rot_vol(ref_vol, data[...,i])
        # or use translated_vol, I'm not sure...

        #translated_rotated_vol = apply_rotations(translated_vol, rot_params)

        # if this is after the first volume, use the previous volume's realignment
        #  parameters as a starting point guess
        if i > 0:
            guess_params = realign_params[i-1,:]
        else:
            guess_params = np.zeros(6)
        resampled_vol, best_params = optimize_map_vol(ref_vol, data[...,i], img.affine, guess_params)

        # add 6 rigid body parameters to array
        #params = np.append(trans_params, rot_params)
        realign_params[i,:] = best_params

        # save new realigned vol in new 4d realigned data array
        realigned_data[...,i] = resampled_vol

        print('Realigned volume:', i)
    # save realigned img with the prefix to the same directory as the input img
    realigned_img = nib.Nifti1Image(data, img.affine)
    nib.save(realigned_img, pjoin(img_dir, prefix + '_map_coords_' + img_name))

    return realigned_img, realign_params

def plot_realignment_parameters(rp, mm = True, degrees = True):

    # input: volume x 6 rigid body motion parameter
    if mm == True:
        rp_adj = rp
        rp_adj[:,0:3] = rp[:,0:3]*3 # voxel sixe to mm
        rp_adj[:,3:6] = np.rad2deg(rp[:,3:6])

    x = range(rp.shape[0])

    f, axarr = plt.subplots(2, sharex=True)
    f.suptitle('Realigment parameters - sub-10159_task-rest_bold.nii', fontsize=14, fontweight='bold')
    axarr[0].plot(x, rp_adj[:,0], color = 'b')
    axarr[0].plot(x, rp_adj[:,1], color = 'g')
    axarr[0].plot(x, rp_adj[:,2], color = 'r')
    axarr[0].set_ylabel('translation parameters (mm)')
    axarr[1].plot(x, rp_adj[:,3], color = 'b')
    axarr[1].plot(x, rp_adj[:,4], color = 'g')
    axarr[1].plot(x, rp_adj[:,5], color = 'r')
    axarr[1].set_ylabel('rotation parameters (degrees)')
    axarr[1].set_xlabel('volumes')
    plt.show()
    plt.savefig('rp_map_coords_sub-10159_task-rest_bold.png', dpi = 200)

    return


img_filename = 'sub-10159_task-rest_bold.nii'
script_dir = dirname(__file__) # directory path where script is located
utils_dir = dirname(script_dir) # directory path with 'tests' folder containing img
code_dir = dirname(utils_dir)
project_dir = dirname(code_dir)
img_path = pjoin(project_dir, 'data', img_filename)

realigned_img, rp = volume_4D_realign(img_path)
plot_realignment_parameters(rp)
np.savetxt('rp_map_coords_prevolguess_sub-10159_task-rest_bold.txt', rp)
