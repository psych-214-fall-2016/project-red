import numpy as np
import dipy
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass, AffineMap, MutualInformationMetric, AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D)
import nibabel as nib


## code_from_dipy.py [Michael]

#resample (dipy); from static and moving, produce new affine


#subject_10159_anat = nib.load('anat/sub-10159_T1w.nii.gz')
#subject_10159_func = nib.load('func/sub-10159_task-rest_bold.nii.gz')
#
# static = subject_10159_anat.get_data()
# static_grid2world = subject_10159_anat.get_affine()
#
#
# moving = np.squeeze(subject_10159_func.get_data()[...,0])
# moving_grid2world = subject_10159_func.get_affine()
#
#
# nbins = 32
# sampling_prop = None
# metric = MutualInformationMetric(nbins, sampling_prop)
#
# level_iters = [10000,1000,100]
#
# sigmas = [3.0,1.0,0.0]
#
# factors = [4,2,1]

"""
basic_resample:
inputs:
static (NIFTI data)
moving (NIFTI data)
output:
resampled (the moving data sampled into the static NIFTI's coordinates)
"""

def basic_resample(static, moving, static_grid2world, moving_grid2world):

    identity = np.eye(4)
    affine_map = AffineMap(identity, static.shape, static_grid2world, moving.shape, moving_grid2world)
    resampled = affine_map.transform(moving)
    return resampled


"""
com_transform:
center of mass transform from dipy; from static and moving, produce center of mass translation affine
inputs:
 static: (nifti data)
 moving: (nifti data)
outputs:
 nbins: number of bins; for the metric
 metric: metric; used to assess fit
 level_iters: array of numbers; how many iterations to run on each pass
 sigmas: array of numbers; amount of smoothing per iteraion (higher means more smoothing)
 factors: array of numbers; how big are the subdivision on each pass (using number of points / factor: '4' means using n/4 samples)
 starting_affine: 4x4 affine matrix; affine to start optimzing from
outputs
 transformed: resampled nifti data
"""

def com_transform(static,moving, static_grid2world, moving_grid2world):

    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)

    return c_of_mass

def translation_transform(static,moving, static_grid2world, moving_grid2world, nbins, sampling_prop, metric, level_iters, sigmas, factors, starting_affine):

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [10000,1000,100]

    sigmas = [3.0,1.0,0.0]

    factors = [4,2,1]

    affreg = AffineRegistration(metric=metric, level_iters=level_iters,sigmas=sigmas,factors=factors)

    transform = TranslationTransform3D()
    params0 = None

    translation = affreg.optimize(static,moving,transform,params0, static_grid2world, moving_grid2world, starting_affine=starting_affine)

    return translation

"""
rigid_transform:
rigid transform from dipy
inputs:
 static: (nifti data)
 moving: (nifti data)
outputs:
 nbins: number of bins; for the metric
 metric: metric; used to assess fit
 level_iters: array of numbers; how many iterations to run on each pass
 sigmas: array of numbers; amount of smoothing per iteraion (higher means more smoothing)
 factors: array of numbers; how big are the subdivision on each pass (using number of points / factor: '4' means using n/4 samples)
 starting_affine: 4x4 affine matrix; affine to start optimzing from
outputs
 transformed: resampled nifti data
"""


def rigid_transform(static, moving, static_grid2world, moving_grid2world, nbins, sampling_prop, metric, level_iters, sigmas, factors, starting_affine):

    transform = RigidTransform3D()
    params0 = None
    affreg = AffineRegistration(metric=metric, level_iters=level_iters,sigmas=sigmas,factors=factors)

    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    return rigid


"""
affine_transform:
affine transform from dipy
inputs:
 static: (nifti data)
 moving: (nifti data)
outputs:
 nbins: number of bins; for the metric
 metric: metric; used to assess fit
 level_iters: array of numbers; how many iterations to run on each pass
 sigmas: array of numbers; amount of smoothing per iteraion (higher means more smoothing)
 factors: array of numbers; how big are the subdivision on each pass (using number of points / factor: '4' means using n/4 samples)
 starting_affine: 4x4 affine matrix; affine to start optimzing from
outputs
 transformed: resampled nifti data
"""


def affine_transform(static, moving, static_grid2world, moving_grid2world, nbins, sampling_prop, metric, level_iters, sigmas, factors, starting_affine):

    transform = AffineTransform3D()
    params0 = None
    affreg = AffineRegistration(metric=metric, level_iters=level_iters,sigmas=sigmas,factors=factors)

    affine = affreg.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=starting_affine)

    return affine
