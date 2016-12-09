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
    """
    correl = np.corrcoef(vol0.ravel(), vol1.ravel())[0, 1]
    return -correl

# Resampling function for any given x, y, z translation
def xyz_trans_vol(vol, x_y_z_trans):
    """ Make a new copy of `vol` translated by `x_y_z_trans` voxels

    inputs:
        vol (3D volume to be translated by the given inputs)
        x_y_z_trans (x_y_z_trans is a sequence or array length 3, containing
        the (x, y, z) translations in voxels.Values in `x_y_z_trans` can be
        positive or negative,and can be floats.)

    outputs:
        trans_vol (new copy of vol translated by x,y,z inputs)
    """
    x_y_z_trans = np.array(x_y_z_trans)
    # [1, 1, 1] says to do no zooming or rotation
    # Resample image using trilinear interpolation (order=1)
    trans_vol = snd.affine_transform(vol, [1, 1, 1], -x_y_z_trans, order=1)
    return trans_vol

# Cost function using xyz translations and the first volume with the given volume
def cost_at_xyz(x_y_z_trans, vol0, vol1):
    """ Give cost function value at xyz translation values `x_y_z_trans`
    """
    # Translate vol1 x,y,z and return the mismatch function value for the given
    #  translation values
    unshifted = xyz_trans_vol(vol1, x_y_z_trans)
    return correl_mismatch(unshifted, vol0)

# Optimization function to obtain best motion parameters and shifted volume
def optimize_trans_vol(vol0, vol1):
    """
    Optimize the translation of vol1 to match vol0

    inputs:
        vol0 - base volume (nth volume)
        vol1 - volume to be translated to best match vol0 (nth + 1 volume)
    outputs:
        best_params: array of [x, y, z] translations to maximize fit between
            vol1 and vol0
        optimized_vol1: vol1 translated by the best parameters to match vol0
    """
    #global vol0, vol1
    best_params = fmin_powell(cost_at_xyz, [0, 0, 0], args = (vol0, vol1,))
    optimized_vol1 = snd.affine_transform(vol1, [1, 1, 1], -best_params)
    return optimized_vol1, best_params
