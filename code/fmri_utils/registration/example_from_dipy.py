from os.path import dirname, join as pjoin

import nibabel as nib
import numpy as np

import dipy
from dipy.viz import regtools
import numpy.linalg as npl
from scipy.ndimage import affine_transform

from fmri_utils.registration.shared import get_data_affine, decompose_rot_mat
from fmri_utils.registration.code_our_version import resample, transform_cmass, transform_rigid, transform_affine
from fmri_utils.registration.code_from_dipy import basic_resample, com_transform, translation_transform, rigid_transform, affine_transform

from dipy.align.imaffine import (transform_centers_of_mass, AffineMap, MutualInformationMetric, AffineRegistration)


## load data

template = nib.load('../../../data/mni_icbm152_t1_tal_nlin_asym_09a.nii')
sub10159_anat = nib.load('../../../data/sub-10159_T1w.nii.gz')

static = template.get_data()
static_affine = template.get_affine()

moving = sub10159_anat.get_data()
moving_affine = sub10159_anat.get_affine()

SCALE = 0.5
SCALE_affine = np.diagflat([SCALE, SCALE, SCALE, 1])

static_shape = np.array(static.shape)
static_resized = resample(np.zeros((static_shape*SCALE).astype('int')), static, np.eye(4), SCALE_affine)
static = static_resized
static_affine = SCALE_affine.dot(static_affine)

moving_shape = np.array(moving.shape)
moving_resized = resample(np.zeros((moving_shape*SCALE).astype('int')), moving, np.eye(4), SCALE_affine)
moving = moving_resized
moving_affine = SCALE_affine.dot(moving_affine)

nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

level_iters = [10000,1000,100]

sigmas = [3.0,1.0,0.0]

factors = [4,2,1]

resampled = basic_resample(static, moving, static_affine, moving_affine)

regtools.overlay_slices(static, resampled, None, 0, "Static", "Moving", "resampled_0.png")
regtools.overlay_slices(static, resampled, None, 1, "Static", "Moving", "resampled_1.png")
regtools.overlay_slices(static, resampled, None, 2, "Static", "Moving", "resampled_2.png")

cmass = com_transform(static,moving, static_affine, moving_affine)
cmass_image = cmass.transform(moving)

regtools.overlay_slices(static, cmass_image, None, 0, "Static", "Moving", "cmass_0.png")
regtools.overlay_slices(static, cmass_image, None, 1, "Static", "Moving", "cmass_1.png")
regtools.overlay_slices(static, cmass_image, None, 2, "Static", "Moving", "cmass_2.png")

translation = translation_transform(static, moving, static_affine, moving_affine, nbins, sampling_prop, metric, level_iters, sigmas, factors, cmass.affine)
translation_image = translation.transform(moving)

regtools.overlay_slices(static, translation_image, None, 0, "Static", "Moving", "translation_0.png")
regtools.overlay_slices(static, translation_image, None, 1, "Static", "Moving", "translation_1.png")
regtools.overlay_slices(static, translation_image, None, 2, "Static", "Moving", "translation_2.png")

rigid = rigid_transform(static, moving, static_affine, moving_affine, nbins, sampling_prop, metric, level_iters, sigmas, factors, translation.affine)
rigid_image = rigid.transform(moving)

regtools.overlay_slices(static, rigid_image, None, 0, "Static", "Moving", "rigid_0.png")
regtools.overlay_slices(static, rigid_image, None, 1, "Static", "Moving", "rigid_1.png")
regtools.overlay_slices(static, rigid_image, None, 2, "Static", "Moving", "rigid_2.png")

affine = affine_transform(static, moving, static_affine, moving_affine, nbins, sampling_prop, metric, level_iters, sigmas, factors, rigid.affine)
affine_image = affine.transform(moving)

regtools.overlay_slices(static, affine_image, None, 0, "Static", "Moving", "affine_0.png")
regtools.overlay_slices(static, affine_image, None, 1, "Static", "Moving", "affine_1.png")
regtools.overlay_slices(static, affine_image, None, 2, "Static", "Moving", "affine_2.png")
