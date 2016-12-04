"""
generate_test_images.py

"""


import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform
from os.path import dirname


T1_10159 = nib.load("../../../data/sub-10159_T1w.nii.gz")

print(T1_10159.affine)


#shift 10159 by translation only

original_translated = affine_transform(T1_10159, np.eye(3), [1,2,3], order=1)
nib.save(original_translated, 'sub-10159_T1w_translated_by_1_2_3.nii')
