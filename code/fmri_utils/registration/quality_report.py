"""
quality check for our code: transform template by known translations & rotations

"""
from os.path import dirname, join as pjoin, abspath
import csv

import numpy as np
import numpy.linalg as npl

import matplotlib.pyplot as plt

import nibabel as nib

from fmri_utils.registration.shared import get_data_affine
from fmri_utils.registration.code_our_version import params2affine, resample, affine_registration, generate_transformed_images, neg_mutual_info, save_affine


### transform MNI template by known translations & rotations

# set paths
MY_DIR = dirname(abspath(__file__))
project_dirname = 'project-red'
idx = MY_DIR.rfind(project_dirname)
project_path = MY_DIR[:(idx+len(project_dirname))]

figs_dir = pjoin(project_path, 'report', 'figures')
template_file = pjoin(project_path, 'data', 'registration_example_files', 'mni_icbm152_t1_tal_nlin_asym_09a_brain.nii.gz')
resampled_file = pjoin(figs_dir, 'mni_icbm152_t1_tal_nlin_asym_09a_brain_changed.nii.gz')

file_prefix = resampled_file.split('/')[-1]
idx = file_prefix.find('.nii')
file_prefix = file_prefix[:idx]

summary_filename = pjoin(figs_dir, file_prefix + '_generated_images_summary.csv')

# set transformation (translation & rotation only)
trans_init = [59, -3, -20] # voxels in x, y, z
rot_init = [0.2, -0.2, 0.5] # radians around x, y, z

# load template
template_data, template_affine = get_data_affine(template_file)

# apply transformation
change_affine = params2affine(trans_init+rot_init)
save_affine(change_affine, figs_dir, 'change_affine.txt')
resampled_data = resample(template_data, template_data, template_affine, template_affine.dot(change_affine))

# save resample dimage
img = nib.Nifti1Image(resampled_data, template_affine)
nib.save(img, resampled_file)

### do affine registration for template and transformed image
SCALE = 1
affine_registration(template_file, resampled_file, 1, figs_dir, 10)
generate_transformed_images(template_file, resampled_file, 1, figs_dir, figs_dir, 'png')

### show MI for each step
labels = ['identical', 'perfect\ninverse', 'trans-\nformed', 'cmass', 'trans-\nlation', 'rigid', 'full\naffine']
neg_MI = [None]*len(labels)

# get mutual info for template and itself (identical)
neg_MI[0] = neg_mutual_info(template_data, template_data)

# get mutual info for template and perfect inverse transform of transfromed image
resampled_data, resampled_affine = get_data_affine(resampled_file)
returned_data = resample(resampled_data, resampled_data, resampled_affine, resampled_affine.dot(npl.inv(change_affine)))
neg_MI[1] = neg_mutual_info(template_data, returned_data)

# load registration step MI values from image summary csv
summary_file = open(summary_filename, 'rU')
summary_dict = {}
for line in csv.reader(summary_file):
    if line[2].find(file_prefix)>-1:
        transform = line[2].split('_')[-1].replace('.txt','')
        summary_dict[transform] = float(line[3])
summary_file.close()

neg_MI[2] = summary_dict['resampled']
neg_MI[3] = summary_dict['cmass']
neg_MI[4] = summary_dict['translation']
neg_MI[5] = summary_dict['rigid']
neg_MI[6] = summary_dict['sheared']

# make fig
fig, ax = plt.subplots(1,1)

ax.bar(range(7),np.array(neg_MI), align = 'center', color = ['red','green']+ ['blue']*5)
ax.plot([-1, 7], [neg_MI[1], neg_MI[1]], 'k--')

ax.set_xticklabels(['']+labels, ha = 'center')
ax.set_ylabel('- MI')

for i, xpos in enumerate(ax.get_xticks()[1:-1]):
    ax.text(xpos,-1.7, '('+str(round(neg_MI[i],3))+')', size = 10,  ha = 'center')
fig.subplots_adjust(bottom=0.2)

ax.set_title('negative mutual information, between template and *')

plt.show()
fig.savefig(resampled_file.replace('.nii.gz', '_MI.png'))
