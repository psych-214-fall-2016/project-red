# - compatibility with Python 2
from __future__ import print_function  # print('me') instead of print 'me'
from __future__ import division  # 1/2 == 0.5, not 0
# - import common modules
import numpy as np  # the Python array package
import numpy.linalg as npl
import matplotlib.pyplot as plt  # the Python plotting package
import scipy.ndimage as snd
from scipy.optimize import fmin_powell
import nibabel as nib

# import of rotations code
#from rotations import x_rotmat, y_rotmat, z_rotmat
from .rotations import x_rotmat, y_rotmat, z_rotmat

# Mismatch function between two volumes
def correl_mismatch(ref_vol, vol1):
    """ Negative correlation between the two images, flattened to 1D

    Parameters
    ----------
    ref_vol: array shape (I, J, K)
        array with data from the reference volume

    vol1: array shape (I, J, K)
        array with data from the volume being transformed to the reference

    Returns
    -------
    correl: float
        correlation coefficient between the reference and transformed volumes

    """
    correl = np.corrcoef(ref_vol.ravel(), vol1.ravel())[0, 1]
    return -correl

# Resampling function for any given volume and the reference (ref_vol)
def apply_coord_mapping(RB_params, ref_vol, vol1, ref_vol_affine):
    """ Resample `ref_vol` to 'vol1' after applying a moving affine based on RB_params

    Parameters
    ----------
    RB_params: array shape (6,)
        array with rigid body transfomations (3 translations, 3 rotations) to be applied

    ref_vol: array shape (I, J, K)
        array with data from the reference volume

    vol1: array shape (I, J, K)
        array with data from the volume being transformed to the reference

    ref_vol_affine: array shape (4, 4)
        image affine mapping for the reference volume

    Returns
    -------
    resampled_vol1: arra shape (I, J, K)
        vol1 resampled according to the rigid body parameters

    """
    # get rotation value sin each direction and make a rotation matrix
    r_x, r_y, r_z = RB_params[3:]
    rot_mat = z_rotmat(r_z).dot(y_rotmat(r_y).dot(x_rotmat(r_x)))
    # get translation values in each direciton
    trans = np.array(RB_params[0:3])
    moving_affine = nib.affines.from_matvec(rot_mat, trans)

    # affine maping bet the moving affine and the affine of the reference volume
    vol1_to_ref_vol = npl.inv(moving_affine).dot(ref_vol_affine)

    # get shape of reference volume
    I, J, K = ref_vol.shape
    # create a mesh coordinate grid
    i_vals, j_vals, k_vals = np.meshgrid(range(I), range(J), range(K), indexing='ij')
    in_vox_coords = np.array([i_vals, j_vals, k_vals])

    np.random.seed(1500)
    #create array of 3*(I,J,K) random numbers between -0.5 and 0.5
    jitter = np.random.uniform(-0.1,0.1, 3*I*J*K)
    # reshape to the same format as in_vox_coords
    jitter = np.reshape(jitter, in_vox_coords.shape)
    # add random noise to voxel coordinates
    in_vox_coords_jittered = in_vox_coords + jitter

    # no jitter:
    #coords_last = in_vox_coords.transpose(1, 2, 3, 0)
    # with jitter:
    coords_last = in_vox_coords_jittered.transpose(1, 2, 3, 0)
    vol1_vox_coords = nib.affines.apply_affine(moving_affine, coords_last)
    coords_first_again = vol1_vox_coords.transpose(3, 0, 1, 2)
    # Resample using map_coordinates
    resampled_vol1 = snd.map_coordinates(vol1, coords_first_again)

    return resampled_vol1


# Cost function using xyz translations and the first volume with the given volume
def cost_at_xyz(RB_params, ref_vol, vol1, ref_vol_affine):
    """ Give cost function value at rigid body transformation values

        Parameters
        ----------
        RB_params: array shape (6,)
            array with rigid body transfomations (3 translations, 3 rotations) to be applied

        ref_vol: array shape (I, J, K)
            array with data from the reference volume

        vol1: array shape (I, J, K)
            array with data from the volume being transformed to the reference

        ref_vol_affine: array shape (4, 4)
            image affine mapping for the reference volume

        Returns
        -------
        correl_mismatch: function
            reference to the mismatch function bewteen the resampled and reference volumes
    """
    # Resample vol1 for x,y,z translations and rotations and return the mismatch
    #  function value for the given rigid body parameters
    resampled_vol = apply_coord_mapping(RB_params, ref_vol, vol1, ref_vol_affine)
    return correl_mismatch(resampled_vol, ref_vol)

# Optimization function to obtain best motion parameters and shifted volume
def optimize_map_vol(ref_vol, vol1, ref_vol_affine, guess_params = np.array([0,0,0,0,0,0])):
    """ Optimize the transofmation of vol1 to best match ref_vol

        Parameters
        ----------
        ref_vol: array shape (I, J, K)
            array with data from the reference volume

        vol1: array shape (I, J, K)
            array with data from the volume being transformed to the reference

        ref_vol_affine: array shape (4, 4)
            image affine mapping for the reference volume

        guess_params: array shape (6,)
            buest guess parameters (3 translations, 3 rotations) for volume realignment, default is 0s


        Returns
        -------
        optimized_vol1: array shape (I, J, K)
            rnew volume with optimized RB parameters applied

        best_params: array shape (6,)
            array of 3 translations and rotations to best match vol1 to reference
    """
    # best guess for rigid body transformations (3 translations, 3 rotations) - starting with 0
    #RB_params_guess = [0, 0, 0, 0, 0, 0]
    best_params = fmin_powell(cost_at_xyz, guess_params, args = (ref_vol, vol1, ref_vol_affine))
    optimized_vol1 = apply_coord_mapping(best_params, ref_vol, vol1, ref_vol_affine)
    best_params = best_params*-1

    return optimized_vol1, best_params
