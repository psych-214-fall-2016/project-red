# I got much of my guidance for this from the NIPYPE Beginners Guide
# Justin Riddle, Dan Lurie, and Zuzanna were all especially helpful,
# as well as the rest of the project group

# *****If registration does not exist or you want deobliqued images after this step,
# take the quotes off of the last step of this script *****

import os
from os.path import dirname, join as pjoin
import nibabel as nib
import numpy as np
from MNI_reorient import MNI_reorient
from skull_strip import structural_skull_strip
from fmri_utils.registration.code_our_version import resample, transform_rigid



print('Running Anatomical Preprocessing: MNI Reorient, Skull Strip, Deoblique.\n')


# start set up for the path name with the anatomical files
ROOTDIR = dirname(__file__)
ROOT_DATA_DIR = pjoin(ROOTDIR, '../../../data')
this_study_dir_id = 'ds000030'

# this is the id for the directory with files used throughout the pipeline/tests
this_study_template_id = 'template_files'

# specific name of the directory with raw T1 images
anatomical_dir_id = 'anat'

# this is an identifier that is consistent across all subjects. It is used
# in order to get a list of all of the subjects
subject_identifier = 'sub'

# this is an identifier found on the end of the anatomical files
structural_identifier = '_T1w.nii.gz'

# create a name for the anatomical results directory
results_dir_id = 'anatomical_results'

# create root names for output files
MNI_reorient_file_root = '_MNI_reorient'
skull_strip_file_root = '_MNI_skull_stripped'
deoblique_root = '_MNI_skull_stripped_deobliqued'


# information needed to deoblique the files using the registration code  that was
# created by Michael and Zuzanna
static_template_image = pjoin(ROOT_DATA_DIR , this_study_template_id, 'mni_icbm152_t1_tal_nlin_asym_09a_skull_stripped.nii.gz')
starting_affine = np.eye(4)

# join the subparts of the anatomical file path
data_dir_string = pjoin(ROOT_DATA_DIR, this_study_dir_id)
subject_names = os.listdir(data_dir_string)


#get a list of each subjects specific numerical identifiers
subjects = [subject for subject in subject_names if subject[0:3] == subject_identifier]
num_subjects = len(subjects)


#get the path to each subjects data directory
subject_dir = []
for Idx in range(num_subjects):
    temp_subject_dir = pjoin(data_dir_string,subjects[Idx])
    subject_dir.append(temp_subject_dir)

#get the path to each subjects anatomical file directory
anatomical_dir = []
for Idx in range(num_subjects):
    temp_anat_dir = pjoin(subject_dir[Idx], anatomical_dir_id)
    anatomical_dir.append(temp_anat_dir)

#get the path to each subjects anatomical file
anatomical_file = []
for Idx in range(num_subjects):
    temp_anat_file = pjoin(anatomical_dir[Idx], subjects[Idx] + structural_identifier)
    anatomical_file.append(temp_anat_file)

#create an anatomical results directory in the same directory as the 'anatomical_dir'
anat_preproc_results_dir = []
for Idx in range(num_subjects):
    path_to_subject_results_dir = pjoin(subject_dir[Idx], results_dir_id)
    if not os.path.isdir(path_to_subject_results_dir):
        os.mkdir(path_to_subject_results_dir)
    anat_preproc_results_dir.append(path_to_subject_results_dir)


# this function creates names/paths for output files.
# this function gets called during anatomcial processing steps
def output_file_generator(in_file, out_file_ID, results_dir, subjectID):
    if in_file[-1] == 'z':
        out_file = pjoin(results_dir, subjectID  + out_file_ID + '.nii.gz')
    else:
        out_file = pjoin(results_dir, subjectID  + out_file_ID + '.nii')
    return(out_file)

# run FSL Reorient2Std in order to make sure the the brain has the proper
# RAS+ orientation
MNI_reorient_results_files = []
for Idx in range(num_subjects):
    input_file = anatomical_file[Idx]
    output_file = output_file_generator(input_file, MNI_reorient_file_root, anat_preproc_results_dir[Idx], subjects[Idx])
    if not os.path.isfile(output_file):
        print('running MNI reorientation for ' + subjects[Idx])
        MNI_reorient(input_file, output_file)
    else:
        print('MNI reorientation already exists for ' + subjects[Idx] + ' : moving to next step')
    MNI_reorient_results_files.append(output_file)


#run FSL Brain Extraction Tool (BET) to get a skull stripped image file.
#To do this, structural_skull_strip is called from skull_strip.py
skull_strip_result_files = []
for Idx in range(num_subjects):
    input_file = MNI_reorient_results_files[Idx]
    output_file = output_file_generator(input_file, skull_strip_file_root, anat_preproc_results_dir[Idx], subjects[Idx])
    if not os.path.isfile(output_file):
        print('running skull strip for ' + subjects[Idx])
        structural_skull_strip(input_file, output_file)
    else:
        print('skull strip already exists for ' + subjects[Idx] + ' : moving to next step')
    skull_strip_result_files.append(output_file)



# Call parts of the registration code to deoblique the brain images
""" **** This step is only necessary if registration is not a future step in the processing pipeline.
         Avoid this step if possible to skip unnecessary resampling of the data.**** """


deobliqued_result_files = []
for Idx in range(num_subjects):
    static_img = nib.load(static_template_image)
    static_data = static_img.get_data()
    static_affine = static_img.affine
    moving_img = nib.load(skull_strip_result_files[Idx])
    moving_data = moving_img.get_data()
    moving_affine = moving_img.affine
    output_file = output_file_generator(skull_strip_result_files[Idx], deoblique_root, anat_preproc_results_dir[Idx], subjects[Idx])
    if not os.path.isfile(output_file):
        print('running rigid transform (deoblique) for ' + subjects[Idx])
        rigid_affine = transform_rigid(static_data, moving_data, static_affine, moving_affine, starting_affine, 10, "all")
        updated_moving_affine = rigid_affine.dot(moving_affine)
        rigid = resample(static_data, moving_data, static_affine, updated_moving_affine)
        img = nib.Nifti1Image(rigid, static_affine)
        nib.save(img, output_file)
    else:
        print('rigid transform (deoblique) already exists for ' + subjects[Idx] + ': moving to next step')
    deobliqued_result_files.append(output_file)

print('\nDone with anatomical MNI reorientation, skull stripping, and deobliquing!')
