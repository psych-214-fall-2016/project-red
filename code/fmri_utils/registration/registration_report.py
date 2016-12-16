"""
demonstration of registration section, to use in report

Default usage in makefile. This will produce figures used in report by loading
results of registration run on the original images (SCALE = 1).

$ python registration_report load 1 sub-10159 sub-10171 sub-10189 sub-10193 sub-10206 sub-10217 sub-10225


You can redo this registration from original skull-stripped MNI and subject T1
images by changing 'load' to 'rerun'. This can take >1 hr/subject, so we
recommend downsampling the images (e.g., SCALE = 0.5) since the results are qualitatively similar.

$ python registration_report rerun 0.5 sub-10159 sub-10171 sub-10189 sub-10193 sub-10206 sub-10217 sub-10225


Note: after registration, we manually identified landmark coordinates for
individual subjects. The subject registration code can be reused for other
subjects, but the script will not attempt to produce the final landmark plots
for subjects outside the original set. The reuse this example
To reuse this example with other subjects, please follow this usage:

$ python registration_report.py instruction scale subj [subj] [...]
    - `version` can be "load" (to reuse saved files) or "rerun" (to do registration again)
    - `scale` is a number >0; 1 = original size (e.g. 0.5 = downsample to 1/2 3D array dimensions)
    - `subj [subj] [...]` is a list of subject ids (must match dir name in project-red/data/ds000030)


"""

import os, sys
from os.path import dirname, join as pjoin, abspath
from shutil import copyfile

import csv
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

import nibabel as nib

from fmri_utils.registration.code_our_version import affine_registration, generate_transformed_images


##### var setup; check necessary files exist #####

# print this error message when command line arguments don't make sense
errormsg = 'expected command line structure:\n\
\n$ python registration_report.py instruction scale subj [subj] [...]\n\
\n- `version` can be "load" (to reuse saved files) or "rerun" (to do registration again)\
\n- `scale` is a number >0; 1 = original size (e.g. 0.5 = downsample to 1/2 3D array dimensions)\
\n- `subj [subj] [...]` is a list of subject ids (must match dir name in project-red/data/ds000030)'

# get abs path for project_dirname
MY_DIR = dirname(abspath(__file__))
project_dirname = 'project-red'
idx = MY_DIR.rfind(project_dirname)

project_path = MY_DIR[:(idx+len(project_dirname))]

# set path for report figures
figs_dir = pjoin(project_path, 'report', 'figures')

# MNI template file to use for registration
static_filename = pjoin(project_path, 'data', 'MNI_template', 'mni_icbm152_t1_tal_nlin_asym_09a_skull_stripped.nii.gz')
if not os.path.exists(static_filename):
    raise RuntimeError('expected static file missing: '+static_filename)

# take instructions, scale, and subject ids from command line
if len(sys.argv) < 4:
    raise RuntimeError('missing args! \n\n'+errormsg)

# `instruction` can be "load" (to display saved results) or "rerun" (to do registration again)
instruction = sys.argv[1]
if instruction not in ['load', 'rerun']:
    raise RuntimeError('wrong instruction! \n\n'+errormsg)

# SCALE must be float >0; SCALE = 1 is original images, smaller values downsample images and speed up registration
try:
    SCALE = float(sys.argv[2])
except ValueError:
    raise RuntimeError('scale arg needs to be numeric! \n\n'+errormsg)

# get subject ids; check that subject data dirs, skull-stripped images, and output dirs exists
subjects = sys.argv[3:]
N = len(subjects)

subject_dirs = [pjoin(project_path, 'data', 'ds000030', s) for s in subjects]
subject_T1s = [pjoin(subject_dirs[i], 'anatomical_results', subjects[i]+'_T1w_skull_stripped.nii.gz') for i in range(N)]
subject_output_dirs = [pjoin(subject_dirs[i], 'registration_results') for i in range(N)]

for i in range(N):
    if not os.path.isdir(subject_dirs[i]):
        print(subject_dirs[i])
        raise RuntimeError('incorrect subject id: '+subjects[i]+'\n\n'+errormsg)
    if not os.path.exists(subject_T1s[i]):
        raise RuntimeError('expected T1 missing: '+subject_T1s[i])
    if not os.path.exists(subject_output_dirs[i]):
        print('created output dir: '+subject_output_dirs[i])
        os.mkdir(subject_output_dirs[i])


##### do registration #####

affine_endings = ['_T1w_skull_stripped_'+A for A in ['resampled','cmass','translation','rigid','sheared']]

if instruction == 'load':
    # check that expected affines exist, raise error

    expected_subject_affines = [pjoin(subject_output_dirs[i], subjects[i]+A+'.txt') for i in range(N) for A in affine_endings]
    for i in range(len(expected_subject_affines)):
        if not os.path.exists(expected_subject_affines[i]):
            raise RuntimeError('trying to load registration restults; expected affine missing: '+expected_subject_affines[i])

elif instruction == 'rerun':
    # produce affines, use SCALE
    for i in range(N):
        print('\nDOING REGISTRATION: '+subjects[i])
        affine_registration(static_filename, subject_T1s[i], SCALE, subject_output_dirs[i], iterations=10)


##### produce *.nii* and *png for registration steps #####

# always display with SCALE=1, even if registration used downsampled images
for i in range(N):
    print('\nGENERATING TRANSFORMED IMAGES: '+subjects[i])
    generate_transformed_images(static_filename, subject_T1s[i], 1, subject_output_dirs[i], subject_output_dirs[i])

subject_sheared_T1s = [pjoin(subject_output_dirs[i], subjects[i]+affine_endings[-1]+'.nii.gz') for i in range(N)]

# copy example pngs to figs_dir for report!
example_png_files = [f.replace('.nii.gz','_2.png') for f in subject_sheared_T1s]

for p in example_png_files:
    img_file = p.split('/')[-1]

    #report images created with:
    copyfile(p, pjoin(figs_dir, img_file))

    #adjust filename to prevent overwrite of report images
    #copyfile(p, pjoin(figs_dir, instruction+'_'+img_file))


##### show manually identified landmarks on transformed T1 images #####

## load landmark location info from coordinate_info.csv
coord_file = open(pjoin(figs_dir,'coordinate_info.csv'),'r')
coord_info = [line for line in csv.reader(coord_file)][1:]

coord_dict = {}
for i in range(len(coord_info)):
    coord_dict[coord_info[i][0]] = coord_info[i]

# add img file paths
coord_dict['MNI'] += [static_filename]
for i in range(N):
    coord_dict[subjects[i]] += [pjoin(subject_output_dirs[i], subject_sheared_T1s[i])]

def str_to_array(txt):
    temp = txt.replace('(','').replace(')','')
    return np.array([float(i) for i in temp.split(', ')])

def save_coords_on_img(subj_line, MNI_line):
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'

    colors = [[0,1,0], [1, 0, 0.5]]
    dot_sets = [subj_line, MNI_line]

    fig, axes = plt.subplots(1,3)
    fig.suptitle(subj_line[0])

    for j in [0,1]:
        img = nib.load(dot_sets[j][8])

        inv_affine = npl.inv(img.affine)
        mat, vec = nib.affines.to_matvec(inv_affine)

        mid_coord = str_to_array(dot_sets[j][2]).dot(mat) + vec

        if j==0:
            data = img.get_data()
            axes[0].imshow(data[int(mid_coord[0]),...].T)
            axes[1].imshow(data[...,int(mid_coord[-1])].T)
            axes[2].imshow(data[...,int(mid_coord[-1])].T)

        axes[0].scatter([mid_coord[1]],[mid_coord[2]], s = 20, c = colors[j])
        axes[1].scatter([mid_coord[0]],[mid_coord[1]], s = 10, c = colors[j])
        axes[2].scatter([mid_coord[0]],[mid_coord[1]], s = 15, c = colors[j])


        for i in range(3,8):
            subj_pt = dot_sets[j][i]

            if subj_pt != '':
                subj_xyz = str_to_array(subj_pt)
                coord = subj_xyz.dot(mat)+vec
                axes[1].scatter([coord[0]],[coord[1]], s = 10, c = colors[j])
                axes[2].scatter([coord[0]],[coord[1]], s = 15, c = colors[j])



    axes[0].set_xlim([0, data.shape[1]])
    axes[0].set_ylim([0, data.shape[2]])

    axes[0].legend([subj_line[0],'MNI'], bbox_to_anchor = (1, 2))

    axes[1].set_xlim([0, data.shape[0]])
    axes[1].set_ylim([0, data.shape[1]])


    axes[2].set_xlim([50, 150])
    axes[2].set_ylim([100, 180])




    return fig

for s in subjects:
    if s in coord_dict:

        f = save_coords_on_img(coord_dict[s], coord_dict['MNI'])
        f.savefig(pjoin(report_dir, s+'.png'))
        plt.show()
    else:
        print('landmarks not identified for this subject; skipping figure generation')

plt.close('all')
