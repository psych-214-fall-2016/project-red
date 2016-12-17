"""
Generate figures from MRF-EM segmentation in the report.

@author: Christine Tseng
"""

import os
from os.path import dirname, join as pjoin

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from fmri_utils.segmentation.mrf_em import mrf_em
from fmri_utils.segmentation.kmeans import kmeans

plt.close('all')

SAVE_FIGS = 0 # 1 to save, 0 to not save

subjects = ['sub-10159', 'sub-10189']

# Current directory
my_dir = dirname(__file__)
# Data directory (project-red/data)
data_dir = pjoin(my_dir,'..','..','..','data')
# Report figures directory (project-red/report/figures)
report_dir = pjoin(data_dir, '..', 'report', 'figures')

# Segment and generate figures
subject_slices = [[130, [60, 90, 90, 110]], [128, [90, 120, 80, 100]]]
for i, s in enumerate(subjects):
    img = nib.load(pjoin(data_dir, 'ds000030', s, 'anat',
                        '%s_T1w_brain.nii.gz'%s))
    data = img.get_data()

    # Slice info
    slice_n = subject_slices[i][0]
    x1, x2, y1, y2 = subject_slices[i][1]
    slice_s = data[x1:x2, y1:y2, slice_n]

    # Segment
    _, labels, maps = mrf_em(slice_s, 0.05, k=2, max_iter=10,
                            scale_range=(100, 400), scale_sigma=50,
                            max_label_iter=10, njobs=2,
                            map_labels=['gray', 'white'])
    # Run kmeans for comparison
    _, klabels, kmaps = kmeans(slice_s.ravel(), k=2, max_iter=10^4,
                             scale_max=400, scale_min=200, map_keys=['gray', 'white'])
    klabels = klabels.reshape(slice_s.shape)

    # Plot segmentation
    fig1 = plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(labels, interpolation="None")
    plt.title('MRF EM')
    plt.subplot(1, 3, 2)
    plt.imshow(slice_s, interpolation="None")
    plt.title('Original')
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(klabels-1), interpolation="None")
    plt.title('kmeans')

    # Save?
    if SAVE_FIGS:
        plt.savefig(pjoin(report_dir, 'mrf_%s_segment.png'%s))

    # Plot probability maps
    fig2 = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(maps['white'], interpolation="None")
    plt.title('White matter')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(maps['gray'], interpolation="None")
    plt.title('Gray matter')
    plt.colorbar()

    # Save?
    if SAVE_FIGS:
        plt.savefig(pjoin(report_dir, 'mrf_%s_pmaps.png'%s))
