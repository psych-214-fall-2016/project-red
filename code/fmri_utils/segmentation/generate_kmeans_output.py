"""
Generates tissue probability maps for three three subjects using k-means and
saves the results.

@author: Christine Tseng
"""

import os
from os.path import dirname, join as pjoin
import numpy as np
import nibabel as nib
from fmri_utils.segmentation.kmeans import kmeans

subjects = ['sub-10159']#, 'sub-10171', 'sub-10189']

# Current directory
my_dir = dirname(__file__)
# Data directory (project-red/data)
data_dir = pjoin(my_dir,'..','..','..','data')

for s in subjects:
    img = nib.load(pjoin(data_dir, 'ds000030', s, 'anat',
                        '%s_T1w_brain.nii.gz'%s))
    data = img.get_data()

    # Segment
    _, klabel, kmap = kmeans(data.ravel())

    # Reshape things
    csf = kmap['csf'].reshape(data.shape)
    gm = kmap['gray'].reshape(data.shape)
    wm = kmap['white'].reshape(data.shape)

    # Save
    nifti_csf = nib.Nifti1Image(csf, affine=img.affine)
    nifti_gm = nib.Nifti1Image(gm, affine=img.affine)
    nifti_wm = nib.Nifti1Image(wm, affine=img.affine)
    nib.nifti1.save(nifti_csf, pjoin(data_dir, 'ds000030', s, 'segmentation_results', 'kmeans_csf.nii'))
    nib.nifti1.save(nifti_gm, pjoin(data_dir, 'ds000030', s, 'segmentation_results', 'kmeans_gm.nii'))
    nib.nifti1.save(nifti_wm, pjoin(data_dir, 'ds000030', s, 'segmentation_results', 'kmeans_wm.nii'))
