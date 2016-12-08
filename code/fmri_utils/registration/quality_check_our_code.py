"""
quality check for our code, using:

A = template
T = transformation

B = transformed A with T
A' = transformed B with inv(T)

T* = found best T with optimization

goal:
can we recover original transformation on image?
what is the best MI match after 2 resampling steps?

"""

from os.path import dirname, join as pjoin

import nibabel as nib
import numpy as np
import numpy.linalg as npl

import dipy
from dipy.viz import regtools
import numpy.linalg as npl
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt

from fmri_utils.registration.shared import get_data_affine, decompose_rot_mat
from fmri_utils.registration.code_our_version import resample, transform_cmass, transform_rigid, transform_affine, mutual_info
from fmri_utils.func_preproc.rotations import x_rotmat, y_rotmat, z_rotmat

## functions

def make_pngs(static, new_img, file):
    for i in range(3):
        regtools.overlay_slices(static, new_img, None, i, "Static", "Moving", file+"_"+str(i)+".png")
    plt.close("all")


## load data
print('load data\n')
template = nib.load('../../../data/mni_icbm152_t1_tal_nlin_asym_09a_brain.nii.gz')

static = template.get_data()
static_affine = template.get_affine()

moving = template.get_data()
moving_affine = template.get_affine()


## set initial transform T
print('set initial transform\n')
rot_init = [0.2, -0.2, 0.5]
trans_init = [59,-3,-20]
rot_mat = z_rotmat(rot_init[2]).dot(y_rotmat(rot_init[1])).dot(x_rotmat(rot_init[0]))
SHIFT_affine = nib.affines.from_matvec(rot_mat, trans_init)
#SHIFT_affine = np.eye(4)

## set downsampling of data (while working to increase speed)
print('set data downsample\n')
SCALE = 0.5
SCALE_affine = nib.affines.from_matvec(np.diagflat([1/SCALE]*3), np.zeros(3))

static_scaled_shape = (np.array(static.shape)*SCALE).astype('int')
moving_scaled_shape = (np.array(moving.shape)*SCALE).astype('int')


## generate starting "static" and "moving" images
static_scaled_affine = static_affine.dot(SCALE_affine)
moving_scaled_affine = moving_affine.dot(SCALE_affine)

static_scaled = resample(np.zeros(static_scaled_shape), static, static_scaled_affine, static_affine)
moving_scaled = resample(np.zeros(moving_scaled_shape), moving, moving_scaled_affine, moving_affine.dot(SHIFT_affine))

static = static_scaled
static_affine = static_scaled_affine

moving = moving_scaled
moving_affine = moving_scaled_affine


## get max MI possible with these images
print('MAX MI for static & static:')
print('...MI = '+str(mutual_info(static, static, 32))+'\n')

mat,vec = nib.affines.to_matvec(npl.inv(SHIFT_affine))
fix_affine = nib.affines.from_matvec(mat,vec/2)
moving_returned = resample(static, moving, static_affine, moving_affine.dot(fix_affine))

print('MAX MI for static & perfect (moving transformed back):')
print('...MI = '+str(mutual_info(static, moving_returned, 32))+'\n')

make_pngs(static, moving_returned, "_perfect")


## resample into template space
print('resample into static space (A_resampled*.png)')
resampled = resample(static, moving, static_affine, moving_affine)

make_pngs(static, resampled, "A_resampled")

print('... MI = '+str(mutual_info(static, resampled, 32))+'\n')


## center of mass transform
print('do center of mass transform (B_cmass*.png)')
cmass_affine = transform_cmass(static, moving, static_affine, moving_affine)
updated_moving_affine = cmass_affine.dot(moving_affine)
cmass = resample(static, moving, static_affine, updated_moving_affine)

make_pngs(static, cmass, "B_cmass")

print('... MI = '+str(mutual_info(static, cmass, 32))+'\n')


## rigid: translation only
print('do translation transfrom (C_translation*.png)')
translation_affine = transform_rigid(static, moving, static_affine, moving_affine, cmass_affine, 10, "translations")
updated_moving_affine = translation_affine.dot(moving_affine)
translation = resample(static, moving, static_affine, updated_moving_affine)

make_pngs(static, translation, "C_translation")

print('... MI = '+str(mutual_info(static, translation, 32))+'\n')


## rigid: translation and rotation
print('do rigid transform (D_rigid*.png)')
rigid_affine = transform_rigid(static, moving, static_affine, moving_affine, translation_affine, 10, "all")
updated_moving_affine = rigid_affine.dot(moving_affine)
rigid = resample(static, moving, static_affine, updated_moving_affine)

make_pngs(static, rigid, "D_rigid")

print('... MI = '+str(mutual_info(static, rigid, 32))+'\n')


## affine: translation, rotation, scaling, and shearing
print('do full affine transform (E_affine*.png)')
shearing_affine = transform_affine(static, moving, static_affine, moving_affine, rigid_affine, 10, "all")
updated_moving_affine = shearing_affine.dot(moving_affine)
sheared = resample(static, moving, static_affine, updated_moving_affine)

make_pngs(static, sheared, "E_affine")

print('... MI = '+str(mutual_info(static, sheared, 32)))
