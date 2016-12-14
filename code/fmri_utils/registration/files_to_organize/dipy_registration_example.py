import numpy as np
import dipy
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass, AffineMap, MutualInformationMetric, AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D)
import nibabel as nib


subject_10159_anat = nib.load('anat/sub-10159_T1w.nii.gz')
subject_10159_func = nib.load('func/sub-10159_task-rest_bold.nii.gz')
#mni_template = nib.load('mni_icbm152_t1_tal_nlin_asym_09a.nii')

#moving = subject_10159_anat.get_data()
#moving_grid2world = subject_10159_anat.get_affine()


#static = np.array(mni_template.get_data())
#static_grid2world = mni_template.get_affine()

static = subject_10159_anat.get_data()
static_grid2world = subject_10159_anat.get_affine()


moving = np.squeeze(subject_10159_func.get_data()[...,0])
moving_grid2world = subject_10159_func.get_affine()


print(static.shape)
print(moving.shape)


identity = np.eye(4)
affine_map = AffineMap(identity, static.shape, static_grid2world, moving.shape, moving_grid2world)
resampled = affine_map.transform(moving)

print(resampled.shape)

regtools.overlay_slices(static, resampled, None, 0, "Static", "Moving", "resampled_0.png")
regtools.overlay_slices(static, resampled, None, 1, "Static", "Moving", "resampled_1.png")
regtools.overlay_slices(static, resampled, None, 2, "Static", "Moving", "resampled_2.png")



c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                      moving, moving_grid2world)

nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

level_iters = [10000,1000,100]

sigmas = [3.0,1.0,0.0]

factors = [4,2,1]

affreg = AffineRegistration(metric=metric, level_iters=level_iters,sigmas=sigmas,factors=factors)

transform = TranslationTransform3D()
params0 = None
starting_affine = c_of_mass.affine


translation = affreg.optimize(static,moving,transform,params0, static_grid2world, moving_grid2world, starting_affine=starting_affine)



transformed = translation.transform(moving)
regtools.overlay_slices(static, transformed, None, 0, "Static", "Transformed", "transformed_trans_0.png")
regtools.overlay_slices(static, transformed, None, 1, "Static", "Transformed", "transformed_trans_1.png")
regtools.overlay_slices(static, transformed, None, 2, "Static", "Transformed", "transformed_trans_2.png")



transform = RigidTransform3D()
params0 = None
starting_affine = translation.affine
rigid = affreg.optimize(static, moving, transform, params0,
                        static_grid2world, moving_grid2world,
                        starting_affine=starting_affine)

transformed = rigid.transform(moving)
regtools.overlay_slices(static, transformed, None, 0,
                        "Static", "Transformed", "transformed_rigid_0.png")
regtools.overlay_slices(static, transformed, None, 1,
                        "Static", "Transformed", "transformed_rigid_1.png")
regtools.overlay_slices(static, transformed, None, 2,
                        "Static", "Transformed", "transformed_rigid_2.png")



transform = AffineTransform3D()
params0 = None
starting_affine = rigid.affine
affine = affreg.optimize(static, moving, transform, params0,
                         static_grid2world, moving_grid2world,
                         starting_affine=starting_affine)

transformed = affine.transform(moving)
regtools.overlay_slices(static, transformed, None, 0,
                        "Static", "Transformed", "transformed_affine_0.png")
regtools.overlay_slices(static, transformed, None, 1,
                        "Static", "Transformed", "transformed_affine_1.png")
regtools.overlay_slices(static, transformed, None, 2,
                        "Static", "Transformed", "transformed_affine_2.png")

