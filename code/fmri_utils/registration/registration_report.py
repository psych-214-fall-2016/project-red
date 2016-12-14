"""
demonstration of registration section, to use in report
"""

import os
from os.path import dirname, join as pjoin
from shutil import copyfile

import csv
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

import nibabel as nib

from fmri_utils.registration.code_our_version import affine_registration, generate_transformed_images


#####


# `static` is the image we want to match (MNI template)
# `moving` is the image we are transforming (subject T1)
# All images are skull-stripped.
#
# There are 12 transformation parameters:
# - 3 translations (along x-, y-, z-axis)
# - 3 rotations (around x-, y-, z-axis)
# - 3 scales (along x-, y-, z-axis)
# - 3 shears (in xy-, xz-, yz-planes)
#
# Our affine registration using the following procedure to fit 12 transformation:
# 0. resample `moving` image in `static` dimensions; visually inspect images look as expected
# 1. find center of mass transform; use as init values for #2
# 2. find best translation (3 free parameters); use as init values for #3
# 3. find best rigid body transformation (6 free parameters); use as init values for #4
# 4. find best full affine transformation (first fit 9 free parameters; use as init values to fit 12 free parameters)

#####

## get affines at each registration step

# subjects used in example
subj_IDs = ['sub-10159', 'sub-10171', 'sub-10189', 'sub-10193', 'sub-10206', 'sub-10217', 'sub-10225']
N = len(subj_IDs)

# current dir, use as navigation reference
MY_DIR = dirname(__file__)

# read data and save outputs (project-red/data)
data_dir = pjoin(MY_DIR,'..','..','..','data')

# individual ouptput dirs (project-red/data/ds000040/sub-#####/registration_results)
subj_output_dirs = [pjoin(data_dir, 'ds000030', s, 'registration_results') for s in subj_IDs]

# will store final transformed T1 -> MNI space filenames here
subj_T1s_inMNI = []

# save figs for report (project-red/report/figures)
report_dir = pjoin(data_dir, '..', 'report', 'figures')

# rescale images for optimization & image generation
SCALE = 1

# MNI template, skull-stripped (DEFAULT)
static_filename = pjoin(data_dir, 'registration_example_files', 'mni_icbm152_t1_tal_nlin_asym_09a_brain.nii.gz')

# inidividual original T1, skull-stripped (DEFAULT)
subj_T1s = [pjoin(data_dir, 'registration_example_files', s+'_T1w_brain.nii.gz') for s in subj_IDs]

# location for saved affines (DEFAULT)
subj_affines_dirs = [pjoin(data_dir, 'registration_example_files') for s in subj_IDs]


''' # default: skip optimization; use saved affines
# alternative: DO OPTIMIZATION, generate affines in `subj_output_dirs`
iterations = 10

# MNI template, skull-stripped (DO OPTIMIZATION)
static_filename = pjoin(data_dir, 'MNI_template', 'mni_icbm152_t1_tal_nlin_asym_09a_skull_stripped.nii.gz')

# inidividual original T1, skull-stripped (DO OPTIMIZATION)
subj_T1s = [pjoin(data_dir, 'ds000030', s, 'anatomical_results', s+'_T1w_skull_stripped.nii.gz') for s in subj_IDs]

# location to save affines (DO OPTIMIZATION)
subj_affines_dirs = subj_output_dirs

# run optimization for each subject
for i in range(N):
    print('REGISTRATION: ' + subj_IDs[i])
    affine_registration(static_filename, subj_T1s[i], SCALE, subj_affines_dirs[i], iterations)
'''

## produce transformed T1 -> MNI *.nii.gz and overlap *.png files

for i in range(N):
    print('GENERATING TRANSFORMED IMAGES: ' + subj_IDs[i])
    #generate_transformed_images(static_filename, subj_T1s[i], SCALE, subj_affines_dirs[i], subj_output_dirs[i])

    img_files = os.listdir(subj_output_dirs[i])
    subj_T1s_inMNI.extend([i for i in img_files if i.find('.nii')>-1 & i.find('sheared')>-1])

## move *.png for sample subject to project-red/report/figures

sample_idx = 0

img_files = os.listdir(subj_output_dirs[sample_idx])
png_files = [i for i in img_files if i.find('.png')>-1]

for p in png_files:
    copyfile(pjoin(subj_output_dirs[sample_idx], p), pjoin(report_dir, p))



##########
# We identified a few landmarks on each transformed image (T1 in MNI space, using final affine transformation):
# 1. Find the z-plane anterior_commissure (x=0, y=0, z=0 on MNI template)
# 2. If landmark visable, find (x,y) coords in this z-plane for:
#    - right anterior insula
#    - right posterior insula
#    - right ventricle top peak
#    - left ventricle top peak
#    - top of white matter (corpus callosum?)

## load landmark location info from coordinate_info.csv
coord_file = open(pjoin(data_dir, 'registration_example_files','coordinate_info.csv'),'r')
coord_info = [line for line in csv.reader(coord_file)][1:]

coord_dict = {}
for i in range(len(coord_info)):
    coord_dict[coord_info[i][0]] = coord_info[i]

# add img file paths
coord_dict['MNI'] += [static_filename]
for i in range(N):
    coord_dict[subj_IDs[i]] += [pjoin(subj_output_dirs[i], subj_T1s_inMNI[i])]

def str_to_array(txt):
    temp = txt.replace('(','').replace(')','')
    return np.array([float(i) for i in temp.split(', ')])

def save_coords_on_img(subj_line, MNI_line):
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'

    colors = [[0,1,0], [1, 0, 0]]
    dot_sets = [subj_line, MNI_line]

    fig, axes = plt.subplots(2,2)
    fig.suptitle(subj_line[0])

    for j in [0,1]:
        img = nib.load(dot_sets[j][8])

        inv_affine = npl.inv(img.affine)
        mat, vec = nib.affines.to_matvec(inv_affine)

        mid_coord = str_to_array(dot_sets[j][2]).dot(mat) + vec

        if j==0:
            data = img.get_data()
            for h in [0,1]:
                axes[h][0].imshow(data[int(mid_coord[0]),...].T)
                axes[h][1].imshow(data[...,int(mid_coord[-1])].T)

        for h in [0,1]:
            axes[h][0].scatter([mid_coord[1]],[mid_coord[2]], s = 6, c = colors[j])
            axes[h][1].scatter([mid_coord[0]],[mid_coord[1]], s = 6, c = colors[j])

        for i in range(3,8):
            subj_pt = dot_sets[j][i]

            if subj_pt != '':
                subj_xyz = str_to_array(subj_pt)
                coord = subj_xyz.dot(mat)+vec
                for h in [0,1]:
                    axes[h][1].scatter([coord[0]],[coord[1]], s = 6, c = colors[j])

    axes[0][0].set_xlim([100, 150])
    axes[0][0].set_ylim([50, 100])

    axes[1][0].set_xlim([0, data.shape[1]])
    axes[1][0].set_ylim([0, data.shape[2]])

    axes[0][0].legend([subj_line[0],'MNI'], bbox_to_anchor = (0.6, 1.1), borderaxespad = 0)

    axes[0][1].set_xlim([50, 150])
    axes[0][1].set_ylim([100, 180])

    axes[1][1].set_xlim([0, data.shape[0]])
    axes[1][1].set_ylim([0, data.shape[1]])


    return fig

for s in subj_IDs:
    f = save_coords_on_img(coord_dict[s], coord_dict['MNI'])
    f.savefig(pjoin(report_dir, s+'.png'))

plt.close('all')


# coord_dict = {}
# for i in range(len(coord_info)):
#     coord_dict[coord_info[i][0]] = coord_info[i][1:]
#
# coord_dict['MNI']+=[static_filename]
# for i in range(N):
#
#     subjID = subj_IDs[i]
#     print(subjID)
#     coord_dict[subjID]+=[pjoin(subj_output_dirs[i],subj_T1s_inMNI[i])]
#
#
# def str_to_list(txt):
#     clean = txt.replace('[','').replace(']','')
#     return [float(i) for i in clean.split(', ')]
#
# def show_coords(subjID, subj_info):
#
#     mm_coords = []
#
#
#     img = nib.load(subj_info[7])
#     data = img.get_data()
#     affine = npl.inv(img.affine)
#
#     mat, vec = nib.affines.to_matvec(affine)
#
#     mid = str_to_list(subj_info[1])
#     z = [mid[-1]]
#
#     mid_coord = np.array(mid).dot(mat) + vec
#     mm_coords+=[mid_coord]
#
#     fig, axes = plt.subplots(1,2)
#     fig.suptitle(subjID)
#
#     axes[0].imshow(data[mid_coord[0],...].T)
#     axes[0].scatter([mid_coord[1]],[mid_coord[2]], c = [0, 1, 0])
#     axes[0].set_xlim([0, data.shape[1]])
#     axes[0].set_ylim([0, data.shape[2]])
#     axes[0].set_xlabel('y-axis')
#     axes[0].set_ylabel('z-axis')
#
#
#     axes[1].imshow(data[...,mid_coord[-1]].T)
#     axes[1].scatter([mid_coord[0]],[mid_coord[1]], c = [0, 1, 0])
#
#
#     for landmark in range(2,7):
#         pt_xy = subj_info[landmark]
#
#         if pt_xy != 'None':
#             pt_xyz = str_to_list(subj_info[landmark])+z
#
#             coord = np.array(pt_xyz).dot(mat) + vec
#             mm_coords+=[coord]
#             axes[1].scatter([coord[0]],[coord[1]], s = 4, c = [0,1,0])
#
#         else:
#             mm_coords+=[None]
#     axes[1].set_xlim([0, data.shape[0]])
#     axes[1].set_ylim([0, data.shape[1]])
#
#     axes[1].set_xlabel('x-axis')
#     axes[1].set_ylabel('y-axis')
#
#     return fig, mm_coords
#
# ALL_mm_coords = {}
# for i in coord_dict:
#     fig, mm_coords = show_coords(i, coord_dict[i])
#     # fig.savefig(pjoin(report_dir, i+'.png'))
#     # plt.close('all')
#
#     ALL_mm_coords[i] = mm_coords
#
# def get_dist(a, b):
#
#     d = np.sqrt(sum((a-b)**2))
#
#     return d
#
# print(ALL_mm_coords)
#
#
# for s in subj_IDs:
#     print(s)
#     MNI_values = ALL_mm_coords['MNI']
#     subj_values = ALL_mm_coords[s]
#
#     for i in range(len(MNI_values)):
#         if subj_values[i]==None:
#             print('N/A')
#         else:
#             print(get_dist(MNI_values[i], subj_values[i]))
