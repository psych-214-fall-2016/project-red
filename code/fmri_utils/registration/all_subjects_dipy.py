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
from os.path import dirname, join as pjoin


## load data


subj_IDs = ['sub-10159', 'sub-10171', 'sub-10189', 'sub-10193', 'sub-10206', 'sub-10217', 'sub-10225']
N = len(subj_IDs)

for i in range(N):

    template = nib.load('../../../data/registration_example_files/mni_icbm152_t1_tal_nlin_asym_09a_brain.nii.gz')

    load_string = '../../../data/ds000030/' + subj_IDs[i] + '/anat/' + subj_IDs[i] + '_T1w.nii.gz'

    subject_anat = nib.load(load_string)

    static = template.get_data()
    static_affine = template.get_affine()

    moving = subject_anat.get_data()
    moving_affine = subject_anat.get_affine()

    SCALE = 1
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

    #regtools.overlay_slices(static, resampled, None, 0, "Static", "Moving", "resampled_0.png")
    #regtools.overlay_slices(static, resampled, None, 1, "Static", "Moving", "resampled_1.png")
    #regtools.overlay_slices(static, resampled, None, 2, "Static", "Moving", "resampled_2.png")

    cmass = com_transform(static,moving, static_affine, moving_affine)
    cmass_image = cmass.transform(moving)

    #regtools.overlay_slices(static, cmass_image, None, 0, "Static", "Moving", "cmass_0.png")
    #regtools.overlay_slices(static, cmass_image, None, 1, "Static", "Moving", "cmass_1.png")
    #regtools.overlay_slices(static, cmass_image, None, 2, "Static", "Moving", "cmass_2.png")

    translation = translation_transform(static, moving, static_affine, moving_affine, nbins, sampling_prop, metric, level_iters, sigmas, factors, cmass.affine)
    translation_image = translation.transform(moving)

    #regtools.overlay_slices(static, translation_image, None, 0, "Static", "Moving", "translation_0.png")
    #regtools.overlay_slices(static, translation_image, None, 1, "Static", "Moving", "translation_1.png")
    #regtools.overlay_slices(static, translation_image, None, 2, "Static", "Moving", "translation_2.png")

    rigid = rigid_transform(static, moving, static_affine, moving_affine, nbins, sampling_prop, metric, level_iters, sigmas, factors, translation.affine)
    rigid_image = rigid.transform(moving)

    #regtools.overlay_slices(static, rigid_image, None, 0, "Static", "Moving", "rigid_0.png")
    #regtools.overlay_slices(static, rigid_image, None, 1, "Static", "Moving", "rigid_1.png")
    #regtools.overlay_slices(static, rigid_image, None, 2, "Static", "Moving", "rigid_2.png")

    affine = affine_transform(static, moving, static_affine, moving_affine, nbins, sampling_prop, metric, level_iters, sigmas, factors, rigid.affine)
    affine_image = affine.transform(moving)



    regtools.overlay_slices(static, affine_image, None, 0, "Static", "Moving", "dipy_output/affine_0_" + subj_IDs[i] + ".png")
    regtools.overlay_slices(static, affine_image, None, 1, "Static", "Moving", "dipy_output/affine_1_" + subj_IDs[i] + ".png")
    regtools.overlay_slices(static, affine_image, None, 2, "Static", "Moving", "dipy_output/affine_2_" + subj_IDs[i] + ".png")
