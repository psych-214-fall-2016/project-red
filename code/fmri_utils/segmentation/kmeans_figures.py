"""
Generates figures in the report for k-means.

@author: Christine Tseng
"""

import os
from os.path import dirname, join as pjoin

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from fmri_utils.segmentation.kmeans import kmeans

plt.close('all')

SAVE_FIGS = 0 # 1 to save, 0 to not save

subjects = ['sub-10159', 'sub-10171', 'sub-10189']

# Current directory
my_dir = dirname(__file__)
# Data directory (project-red/data)
data_dir = pjoin(my_dir,'..','..','..','data')
# Report figures directory (project-red/report/figures)
report_dir = pjoin(data_dir, '..', 'report', 'figures')

# Segmentation for subjects
for s in subjects:
    img = nib.load(pjoin(data_dir, 'ds000030', s, 'anat',
                        '%s_T1w_brain.nii.gz'%s))
    data = img.get_data()
    n_slices = data.shape[-1]
    slice_s = data[..., np.floor(n_slices/2).astype(int)] # middle slice

    # Segment
    _, klabel, kmap = kmeans(slice_s.ravel())

    # Plots
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 4, 1)
    plt.imshow(klabel.reshape(slice_s.shape))
    plt.title('Segmented slice')
    plt.subplot(1, 4, 2)
    plt.imshow(kmap['csf'].reshape(slice_s.shape), cmap='gray')
    plt.title('CSF')
    plt.subplot(1, 4, 3)
    plt.imshow(kmap['gray'].reshape(slice_s.shape), cmap='gray')
    plt.title('Gray matter')
    plt.subplot(1, 4, 4)
    im = plt.imshow(kmap['white'].reshape(slice_s.shape), cmap='gray')
    plt.title('White matter')

    # Save?
    if SAVE_FIGS:
        plt.savefig(pjoin(report_dir, 'kmeans_%s.png'%s))

plt.draw()
plt.show()
