import nibabel as nib
import os
from os.path import dirname, join as pjoin
import matplotlib.pyplot as plt

# get the directory paths set up
ROOTDIR = dirname(__file__)
ROOT_DATA_DIR = pjoin(ROOTDIR, '../../../data/ds000030')
subject_names = os.listdir(ROOT_DATA_DIR)
# get subject identification information
subjects = [subject for subject in subject_names if subject[0:3] == 'sub']
num_subjects = len(subjects)

# get the path to the original T1 image directory
anat_original_dir = []
for Idx in range(num_subjects):
    temp_subject_dir = pjoin(ROOT_DATA_DIR,subjects[Idx], 'anat')
    anat_original_dir.append(temp_subject_dir)

# get the path to the anatomical results directory
anat_results_dir = []
for Idx in range(num_subjects):
    temp_subject_dir = pjoin(ROOT_DATA_DIR,subjects[Idx], 'anatomical_results')
    anat_results_dir.append(temp_subject_dir)

# get the path name of the original T1 images
# og = original
og_results_files = []
for Idx in range(num_subjects):
    temp_result_file = pjoin(anat_original_dir[Idx], subjects[Idx] + '_T1w.nii.gz')
    og_results_files.append(temp_result_file)


# get the path name of the MNI reoriented images
# MNI = MNI_reorient
MNI_results_files = []
for Idx in range(num_subjects):
    temp_result_file = pjoin(anat_results_dir[Idx], subjects[Idx] + '_MNI_reorient.nii.gz')
    MNI_results_files.append(temp_result_file)

# get the path name of the skull stripped images
# ss = skull_stripped
ss_results_files = []
for Idx in range(num_subjects):
    temp_result_file = pjoin(anat_results_dir[Idx], subjects[Idx] + '_MNI_skull_stripped.nii.gz')
    ss_results_files.append(temp_result_file)

# get the path name of the deobliqued images (if this step was processed)
# db = deobliqued
db_results_files = []
for Idx in range(num_subjects):
    temp_result_file = pjoin(anat_results_dir[Idx], subjects[Idx] + '_MNI_skull_stripped_deobliqued.nii.gz')
    db_results_files.append(temp_result_file)

# get data information from each of the different images to be viewed
og_img = nib.load(og_results_files[1])
og_data = og_img.get_data()

MNI_img = nib.load(MNI_results_files[1])
MNI_data = MNI_img.get_data()

ss_img = nib.load(ss_results_files[1])
ss_data = ss_img.get_data()

db_img = nib.load(db_results_files[1])
db_data = db_img.get_data()

fig , ax = plt.subplots(4)



ax[0].imshow(og_data[:,:,150], cmap='gray')

ax[1].imshow(MNI_data[:,:,150], cmap='gray')

ax[2].imshow(ss_data[:,:,150], cmap='gray')

ax[3].imshow(db_data[:,:,90], cmap='gray')
plt.show()
