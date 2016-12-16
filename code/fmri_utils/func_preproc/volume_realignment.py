""" Volume realignment script
"""

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
import sys
import argparse


# import translations and rotation optimization functions
from fmri_utils.func_preproc.translations import optimize_trans_vol
from fmri_utils.func_preproc.optimize_rotations import optimize_rot_vol, apply_rotations
from fmri_utils.func_preproc.optimize_map_coordinates import optimize_map_vol, apply_coord_mapping


def volume_4D_realign(img_path, reference = 0, smooth_fwhm = 0, jitter = 0.0, prefix = 'r', drop_vols = 0, guess_params = np.zeros(6), mean_ref = np.zeros((64,64,30))):
    """ Realign volumes obtained from a 4D Nifti1Image file

    Input
    ----------
    img_path: string
        filepath of the 4D .nii image for which volumes will be realigned

    reference: int
        int with value 0 (default) or 1, indicating to use the first (0) or middle (1) volume as the reference

    smooth_fwhm: int
        value (in mm) of the FWHM used to smooth the image data before realignment

    jitter: float
        amount of jitter optional to add random noise for realignment

    prefix: string
        short string (default = 'r') added as the prefix for the returned 4D nii image

    drop_vols: int
        index of the volume before which all volumes will be dropped (defualt = 0)

    mean_ref: array shape (I x J x K)
        Mean functional volume (if provided this will be the reference)

    Output
    -------
    realigned_img: .nii file
        Nifti1Image containing the realigned volumes

    realign_params: array shape (volumes x 6)
        2D numpy array containign the 6 rigid body transformation values for each volume
    """
    img_dir, img_name = os.path.split(img_path)

    img = nib.load(img_path)
    data = img.get_data()

    # Dropping volumes, specified by input of drop_vols
    data = data[...,drop_vols:]

    # smooth image, if designated
    if smooth_fwhm > 0:
        fwhm = smooth_fwhm
        img_smooth = proc.smooth_image(img, fwhm)
        img_optimize = img_smooth
        data_optimize = img_smooth.get_data()
    else:
        img_optimize = img
        data_optimize = data


    n_vols = data.shape[-1]
    print('Number of volumes to realign:', n_vols)

    # set whether the reference should be the first volume or middle volume
    if reference == 0:
        ref_vol = data_optimize[...,0]
    elif reference == 1:
        mid_index = int(n_vols/2)
        mid_vol = data_optimize[...,mid_index]
        ref_vol = mid_vol

    if sum(sum(sum(mean_ref))) > 0:
        ref_vol = mean_ref


    # array to which the realignment parameters for each of the 6 rigid body parameters
    #  will be added for each volume
    realign_params = np.zeros((n_vols, 6))
    realigned_data = np.zeros((data.shape))
    for i in range(n_vols):
        # Use either zeros (default) or inputted parameters tog uess starting point for volume realignment
        if len(guess_params.shape) > 1:
            guess_params_vol = guess_params[i,:]*(-1)
        else:
            guess_params_vol = guess_params

        # getting best parameters for the i-th volume
        best_params = optimize_map_vol(ref_vol, data_optimize[...,i], img_optimize.affine, guess_params_vol, jitter = jitter)

        # resampling using params determined above
        resampled_vol = apply_coord_mapping(best_params, ref_vol, data[...,i], img.affine)

        # add 6 rigid body parameters to array
        #params = np.append(trans_params, rot_params)
        realign_params[i,:] = best_params

        print('Realigned volume:', i)

        # place new realigned vol in new 4d realigned data array
        realigned_data[...,i] = resampled_vol

        # save realigned img with the prefix to the same directory as the input img
        realigned_img = nib.Nifti1Image(data, img.affine)

    return realigned_img, realign_params

def plot_realignment_parameters(rp, mm = True, degrees = True, voxel_size = 3.0, title = 'Realignment parameters'):
    """ Realign volumes obtained from a 4D Nifti1Image file

    Input
    ----------
    rp: array shape (volumes x 6)
        2D numpy array containing the 6 rigid body transformation values for each volume

    mm: boolean (default = True)
        plot the translation values in mm, not voxels

    degrees: boolean (default = True)
        plot the translation values in degrees, not radians

    voxel_size: float (default = 3.0)
        size of the voxels in the 3D Nifti file of interest

    title: string
        title fo the plots fo the realignment parameters

    Output
    -------

    f: matplotlib figure object

    """
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

    # Make legends outside of plots
    box_0 = axarr[0].get_position()
    axarr[0].set_position([box_0.x0, box_0.y0, box_0.width * 0.8, box_0.height])
    axarr[0].legend(('X','Y', 'Z'), loc='center left', bbox_to_anchor=(1, 0.5))

    box_1 = axarr[1].get_position()
    axarr[1].set_position([box_1.x0, box_1.y0, box_1.width * 0.8, box_1.height])
    axarr[1].legend(('r_x','r_y', 'r_z'), loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

    return f

def run_realignment(img_name, smooth_fwhm = 5, num_pass = 1, prefix = 'r_'):
    """ Realign volumes obtained from a 4D Nifti1Image file

        Input
        ----------
        img_name: string
            filename of the .nii image for which volumes will be realigned
            (from the 'data' directory in the project-red repository)

        smooth_fwhm: int (value should be 0 to 7mm, default = 5)
            value (in mm) of the FWHM used to smooth the image data before realignment
            in the case of a 2-pass suggestion, first pass is at 8mm, second at specified smoothing

        num_pass: int (default = 1, option for 2)
            If == 1, do one pass with 5mm smoothing
            If == 2, do two passes, first at 8mm, second at 4mm, using parameters from first pass as guess for second

        prefix: string (default = 'r')

        Output
        -------

        Saved plot of volumes vs. the 6 rigid-body realignment parameters

        Saved text file of volumes vs. the 6 rigid-body realignment parameters

        Saved 4D .nii image with resampled, realigned volumes

    """
    script_dir = dirname(__file__) # directory path where script is located
    utils_dir = dirname(script_dir) # directory path with 'tests' folder containing img
    code_dir = dirname(utils_dir)
    project_dir = dirname(code_dir)
    img_filename = img_name
    img_path = pjoin(project_dir, 'data', img_filename)

    if num_pass == 1:
        realigned_img, rp = volume_4D_realign(img_path, smooth_fwhm = smooth_fwhm, jitter = 0.1, prefix = prefix)
    elif num_pass == 2:
        realigned_img, rp_1 = volume_4D_realign(img_path, smooth_fwhm = 8, jitter = 0.1, prefix = prefix)
        mean_vol = get_mean_vol(realigned_img)
        realigned_img_2, rp = volume_4D_realign(img_path, smooth_fwhm = smooth_fwhm, jitter = 0.1, prefix = prefix, guess_params = (rp_1*(-1)), mean_ref = mean_vol)


    graph_name = pjoin(project_dir, 'report', 'figures', prefix + os.path.splitext(img_name)[0])
    fig = plot_realignment_parameters(rp, title = ('Realignment parameters - ' + img_name))
    fig.savefig(graph_name + '.png', dpi = 200)
    np.savetxt(graph_name  + '.txt', rp)

    nib.save(realigned_img, pjoin(project_dir, 'data', prefix + img_filename))

    return

def get_mean_vol(img):
    """ Realign volumes obtained from a 4D Nifti1Image file

    Input
    ----------
    img: Nifti1Image
        img for which the mean volume will be calculated

    Output
    -------
    mean_vol: array shape (I x J x K)

    """

    img = nib.load(img)
    data = img.get_data()

    mean_vol = np.mean(data, axis = -1)

    return mean_vol


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", type = str, required = True)
    parser.add_argument("-s", "--smooth", type = int, required = False, default = 5)
    parser.add_argument("-n", "--num_pass", type = int, required = False, default = 1)
    parser.add_argument("-p", "--prefix", type = str, required = False, default = 'r_')

    args = parser.parse_args()
    print(args)
    #print args.string
    #print args.integer

    # if len(sys.argv) < 2:
    #     raise RuntimeError('Missing filename!')
    #
    # if len(sys.argv) == 2:
    #     scan = sys.argv[1] # filename of scan in the data folder
    #     run_realignment(scan)
    #
    # if len(sys.argv) == 3:
    #     smooth = int(sys.argv[2]) # FWHM smoothing (in mm, 0-7)
    #     num_pass = int(sys.argv[3]) # 1 or 2 pass realignment
    #     prefix = sys.argv[4] # string to be placed in front of filenames

    run_realignment(args.file, smooth_fwhm = args.smooth, num_pass = args.num_pass, prefix = args.prefix)


if __name__ == '__main__':
    main()
