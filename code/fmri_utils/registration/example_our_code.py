from os.path import dirname, join as pjoin

import nibabel as nib
import numpy as np
import numpy.linalg as npl

import dipy
from dipy.viz import regtools
import numpy.linalg as npl
from scipy.ndimage import affine_transform

from fmri_utils.registration.shared import get_data_affine, decompose_rot_mat
from fmri_utils.registration.code_our_version import resample, transform_cmass, transform_rigid, transform_affine


## load data
print('loading data...')
template = nib.load('../../../data/mni_icbm152_t1_tal_nlin_asym_09a_brain.nii.gz')
sub10159_anat = nib.load('../../../data/sub-10159_T1w_brain.nii.gz')

static = template.get_data()
static_affine = template.get_affine()

moving = sub10159_anat.get_data()
moving_affine = sub10159_anat.get_affine()

## downsample data while working to increase speed, where:
#       original shape = (X,Y,Z)
#       new shape = SCALE*(X,Y,Z)
print('downsampling data...')
SCALE = 0.5
SCALE_affine = nib.affines.from_matvec(np.diagflat([1/SCALE]*3), np.zeros(3))

static_scaled_shape = (np.array(static.shape)*SCALE).astype('int')
moving_scaled_shape = (np.array(moving.shape)*SCALE).astype('int')
static_scaled_affine = static_affine.dot(SCALE_affine)
moving_scaled_affine = moving_affine.dot(SCALE_affine)

static_scaled = resample(np.zeros(static_scaled_shape), static, static_scaled_affine, static_affine)
moving_scaled = resample(np.zeros(moving_scaled_shape), moving, moving_scaled_affine, moving_affine)

static = static_scaled
static_affine = static_scaled_affine

moving = moving_scaled
moving_affine = moving_scaled_affine

## resample into template space
print('working on resampled*.png')
resampled = resample(static, moving, static_affine, moving_affine)

regtools.overlay_slices(static, resampled, None, 0, "Static", "Moving", "resampled_0.png")
regtools.overlay_slices(static, resampled, None, 1, "Static", "Moving", "resampled_1.png")
regtools.overlay_slices(static, resampled, None, 2, "Static", "Moving", "resampled_2.png")


## center of mass transform
print('working on cmass*.png')
cmass_affine = transform_cmass(static, moving, static_affine, moving_affine)
updated_moving_affine = cmass_affine.dot(moving_affine)
cmass = resample(static, moving, static_affine, updated_moving_affine)

regtools.overlay_slices(static, cmass, None, 0, "Static", "Moving", "cmass_0.png")
regtools.overlay_slices(static, cmass, None, 1, "Static", "Moving", "cmass_1.png")
regtools.overlay_slices(static, cmass, None, 2, "Static", "Moving", "cmass_2.png")


## rigid: translation only
print('working on translation*.png')
translation_affine = transform_rigid(static, moving, static_affine, moving_affine, cmass_affine, 10, "translations")
updated_moving_affine = translation_affine.dot(moving_affine)
translation = resample(static, moving, static_affine, updated_moving_affine)

regtools.overlay_slices(static, translation, None, 0, "Static", "Moving", "translation_0.png")
regtools.overlay_slices(static, translation, None, 1, "Static", "Moving", "translation_1.png")
regtools.overlay_slices(static, translation, None, 2, "Static", "Moving", "translation_2.png")

## rigid: translation and rotation
print('working on rigid*.png')
rigid_affine = transform_rigid(static, moving, static_affine, moving_affine, translation_affine, 10, "all")
updated_moving_affine = rigid_affine.dot(moving_affine)
rigid = resample(static, moving, static_affine, updated_moving_affine)

regtools.overlay_slices(static, rigid, None, 0, "Static", "Moving", "rigid_0.png")
regtools.overlay_slices(static, rigid, None, 1, "Static", "Moving", "rigid_1.png")
regtools.overlay_slices(static, rigid, None, 2, "Static", "Moving", "rigid_2.png")

# ## affine: translation, rotation, and scaling
# print('working on scaled*.png')
# scaling_affine = transform_affine(static, moving, static_affine, moving_affine, rigid_affine, 10, "scales")
# updated_moving_affine = scaling_affine.dot(moving_affine)
# scaled = resample(static, moving, static_affine, updated_moving_affine)
#
# regtools.overlay_slices(static, scaled, None, 0, "Static", "Moving", "scaled_0.png")
# regtools.overlay_slices(static, scaled, None, 1, "Static", "Moving", "scaled_1.png")
# regtools.overlay_slices(static, scaled, None, 2, "Static", "Moving", "scaled_2.png")

## affine: translation, rotation, scaling, and shearing
print('working on sheared*.png')
# shearing_affine = transform_affine(static, moving, static_affine, moving_affine, scaling_affine, 10, "all")
shearing_affine = transform_affine(static, moving, static_affine, moving_affine, rigid_affine, 10, "all")
updated_moving_affine = shearing_affine.dot(moving_affine)
sheared = resample(static, moving, static_affine, updated_moving_affine)

regtools.overlay_slices(static, sheared, None, 0, "Static", "Moving", "sheared_0.png")
regtools.overlay_slices(static, sheared, None, 1, "Static", "Moving", "sheared_1.png")
regtools.overlay_slices(static, sheared, None, 2, "Static", "Moving", "sheared_2.png")
