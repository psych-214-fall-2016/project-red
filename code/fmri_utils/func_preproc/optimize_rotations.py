# - compatibility with Python 2
from __future__ import print_function  # print('me') instead of print 'me'
from __future__ import division  # 1/2 == 0.5, not 0
# - import common modules
import numpy as np  # the Python array package
import matplotlib.pyplot as plt  # the Python plotting package
import scipy.ndimage as snd
from scipy.optimize import fmin_powell
# - set gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
# Tell numpy to print numbers to 4 decimal places only
np.set_printoptions(precision=4, suppress=True)

""" Common definitions for the following functions:

    vol0: the volume to which the next volume is being transformed to
    - volume 0 of the series
    - is a volume which is presumed to be the

    vol1: the volume being transformed to match vol1
    - starts at the volume 1 of the series
    - is shifted relative to vol0
"""

# Mismatch function between two slices or volumes
def correl_mismatch(vol0, vol1):
    """ Negative correlation between the two images, flattened to 1D

    inputs:
        vol0, vol1 (2D slices or 3D volumes)

    outputs:
        negative correlation coefficient between vol0, vol1

        *Same mismatch function as that of the translations code
    """
    correl = np.corrcoef(vol0.ravel(), vol1.ravel())[0, 1]
    return -correl

# Resampling function for any given x, y, z rotation
def apply_rotations(vol, rotations):
        """ Make a new copy of `vol` rotated by `rotations` along x,y,z (pitch, yaw, roll?)

        inputs:
            vol (3D volume to be rotated by the given inputs)
            rotations (rotation is a sequence or array length 3, containing
            the (x, y, z) rotations in degrees. Values in `rotations` can be
            positive or negative,and can be floats.)

        outputs:
            rot_vol (new copy of vol rotated by x,y,z inputs)
        """
    r_x, r_y, r_z = rotations
    rotation_matrix = z_rotmat(r_z).dot(y_rotmat(r_y).dot(x_rotmat(r_x)))
    # apply rotations with affine_transform to make new image
    rot_vol = snd.affine_transform(vol, rotation_matrix)
    # return new image
    return rot_vol

# Cost function using xyz translations and the first volume with the given volume
def cost_at_xyz(rotations, vol0, vol1):
    """ Give cost function value at xyz rotation values `rotations`
    """
    # Translate vol1 x,y,z and return the mismatch function value for the given
    #  translation values
    derotated = apply_rotations(vol1, rotations)
    return correl_mismatch(derotated, vol0)

# Optimization function to obtain best motion parameters and shifted volume
def optimize_rot_vol(vol0, vol1):
    """
    Optimize the rotation of vol1 to match vol0

    inputs:
        vol0 - base volume (nth volume)
        vol1 - volume to be rotated to best match vol0 (nth + 1 volume)
    outputs:
        best_params: array of [r_x, r_y, r_z] rotations to maximize fit between
            vol1 and vol0
        optimized_vol1: vol1 rotated by the best parameters to match vol0
    """
    #global vol0, vol1
    best_params = fmin_powell(cost_at_xyz, [0, 0, 0], args = (vol0, vol1,))
    r_x, r_y, r_z = best_params
    rotation_matrix = z_rotmat(r_z).dot(y_rotmat(r_y).dot(x_rotmat(r_x)))
    optimized_vol1 = apply_rotations(vol1, best_params)
    return optimized_vol1, best_params
    #unsure about whet to use the negative here or not
