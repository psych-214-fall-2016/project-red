#got most of my guidance for this from the NIPYPE Beginners Guide

import os
from os.path import join as opj
from MNI_reorient import MNI_reorient
from skull_strip import structural_skull_strip
#from fmri_utils.registration.code_our_version import resample, transform_rigid

#import deoblique

print('Running Anatomical Preprocessing: Skull Strip, MNI Reorientation')
#set up all of the directories
ROOTDIR = '/Users/despolab/CMF_Files/'
data_dir_id = 'data/ds000030/'
subject_identifier = 'sub'
anatomical_dir_id = 'anat'
structural_identifier = '_T1w.nii.gz'
results_dir_id = 'anatomical_results'

#create names for output files
MNI_reorient_file_root = '_MNI_reorient'
skull_strip_file_root = '_MNI_skull_strip'


data_dir_string = opj(ROOTDIR, data_dir_id)
data_dir = os.listdir(data_dir_string)

#get a list of all of the subject identifiers
subjects = [subject for subject in data_dir if subject[0:3] == subject_identifier]
num_subjects = len(subjects)


#get the path the each subjects directory
subject_dir = []
for Idx in range(num_subjects):
    temp_subject_dir = opj(data_dir_string,subjects[Idx])
    subject_dir.append(temp_subject_dir)

#get the path the each subjects anatomical directory
anatomical_dir = []
for Idx in range(num_subjects):
    temp_anat_dir = opj(subject_dir[Idx], anatomical_dir_id)
    anatomical_dir.append(temp_anat_dir)

#get the path the each subjects anatomical file
anatomical_file = []
for Idx in range(num_subjects):
    temp_anat_file = opj(anatomical_dir[Idx], subjects[Idx] + structural_identifier)
    anatomical_file.append(temp_anat_file)

#create an anatomical results directory in the same file as the 'anatomical_dir'
anat_preproc_results_dir = []
for Idx in range(num_subjects):
    path_to_subject_results_dir = opj(subject_dir[Idx], results_dir_id)
    if not os.path.isdir(path_to_subject_results_dir):
        os.mkdir(path_to_subject_results_dir)
    anat_preproc_results_dir.append(path_to_subject_results_dir)



def output_file_generator(in_file, out_file_ID, results_dir, subjectID):
    if in_file[-1] == 'z':
        out_file = opj(results_dir, subjectID  + out_file_ID + '.nii.gz')
    else:
        out_file = opj(results_dir, subjectID  + out_file_ID + '.nii')
    return(out_file)

MNI_reorient_results_files = []
for Idx in range(num_subjects):
    input_file = anatomical_file[Idx]
    output_file = output_file_generator(input_file, MNI_reorient_file_root, anat_preproc_results_dir[Idx], subjects[Idx])
    if not os.path.isfile(output_file):
        print('running MNI reorientation for ' + subjects[Idx])
        MNI_reorient(input_file, output_file)
    else:
        print('MNI reorientation already exists for ' + subjects[Idx] + ' : skipping this subject')
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
        print('skull strip already exists for ' + subjects[Idx] + ' : skipping this subject')
    skull_strip_result_files.append(output_file)

#get static/moving_affine by doing nib.load affine) img.get_affine)
"""print('do rigid transform (D_rigid*.png)')
rigid_affine = transform_rigid(static, moving, static_affine, moving_affine, translation_affine, 10, "all")
updated_moving_affine = rigid_affine.dot(moving_affine)
rigid = resample(static, moving, static_affine, updated_moving_affine))
#for Idx in range(num_subjects):"""
