# Set up our usual routines and configuration
import os
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
# - set gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
import nibabel as nib

from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)


def deoblique(skull_stripped_image):
    moving_img = nib.load(skull_stripped_image)
    template_img = nib.load('mni_icbm152_t1_tal_nlin_asym_09a_masked_222.nii')

    moving_data = moving_img.get_data()
    moving_affine = moving_img.affine
    template_data = template_img.get_data()
    template_affine = template_img.affine


    identity = np.eye(4)
    affine_map = AffineMap(identity,
                           template_data.shape, template_affine,
                           moving_data.shape, moving_affine)
    resampled = affine_map.transform(moving_data)

    # The mismatch metric
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    # The optimization strategy
    level_iters = [10, 10, 5]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    translation = affreg.optimize(template_data, moving_data, transform, params0,
                                  template_affine, moving_affine)

    transform = RigidTransform3D()
    rigid = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=translation.affine)


    rigid.affine  #######
    return rigid 
