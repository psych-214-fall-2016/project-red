"""
general functions, useful for other registration methods and tests

"""

import nibabel as nib
import numpy as np
from fmri_utils.func_preproc.rotations import x_rotmat, y_rotmat, z_rotmat


def get_data_affine(fname):
    """ get data and affine from filename

    Parameters
    ----------
    fname : str
        *.nii or *.nii.gz file name

    Returns
    -------
    data : array shape (I, J, K) or (I, J, K, T)
        array with 3D or 4D data from image file
    affine : array shape (4, 4)
        affine for image file
    """
    img = nib.load(fname)
    data = img.get_data()
    affine = img.affine

    return (data, affine)


def make_rot_mat(rotations):
    """ make rotation matrix from radian params

    Parameters
    ----------
    rotations : list
        3 radian values, e.g. [pi, 0, -pi]

    Returns
    -------
    rot_mat : array shape (3, 3)
        rotation matrix
    """
    r_x,r_y,r_z = rotations
    rot_mat = z_rotmat(r_z).dot(y_rotmat(r_y)).dot(x_rotmat(r_x))
    return rot_mat



def decompose_rot_mat(R):
    """ get radian params from rotation matrix
        NOTE:
            will return params in ranges:
            r_x -> (-pi, pi)
            r_y -> (-pi/2, pi/2)
            r_z -> (-pi, pi)

    Parameters
    ----------
    rot_mat : array shape (3, 3)
        rotation matrix

    Returns
    -------
    rotations : list
        3 radian values, e.g. [pi, 0, -pi]

    """
    r_x = np.arctan2(R[2,1], R[2,2])
    r_y = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    r_z = np.arctan2(R[1,0], R[0,0])

    return r_x, r_y, r_z
