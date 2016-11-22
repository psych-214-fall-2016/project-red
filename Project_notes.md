## Project info

- I figured this could be a nice forum to jot down some project notes
- Please feel free to add and expand, tried to recall plan from our last meeting

## Data links:
https://openfmri.org/dataset/ds000030/
http://biorxiv.org/content/early/2016/06/19/059733

## Outline of preprocessing:
http://windstalker.pbworks.com/f/1343426443/02_Preprocessing_FIL2011May.png
- realignment
- motion correction
- coregistration
- segmentation
- spatial normalization
- smoothing
(just reiterating the above diagram, we may need more or fewer steps)

## Project overview:
- Implement and build checks for a preprocessing pipeline of a resting-state fMRI
dataset.
- Once checks have been completed, analyze the resting-state data from the different
patient and control groups in the study.
  - Start to look at different graph theory measures

## Workflow:
- Split up steps of preprocessing among different group members to research each
respective step and brainstorm possible checks to ensure the step works and
is appropriate for the data?
- [Workflow example from CPAC](http://fcp-indi.github.io/docs/user/running.html)

Articles about resting-state preprocessing and analysis:

## Breaking up the work
- Registration (Michael and Zuzana)
- Segmentation (Christine)
- Functional Preproc (Jacob)
- Anatomical Preproc (Chris)
- Nuisance (TBD; Jacob + Christine?)

## Pipeline Overview

Pipeline inputs:
- T1
- EPI
- MNI Template
- Population Priors

All inputs and outputs are NIfTI files unless otherwise specified.

### Anatomical Preprocessing

Software:
- ANTs (Brain Extraction)
- AFNI

In:
- Raw T1
- Population Priors

Out: 
- Deobliqued, RPI reoriented, bias-corrected, skull-stripped T1
- Deobliqued, RPI reoriented, bias-corrected, whole-head T1
- Brain mask.

### Functional Preprocessing

Software:
- fslmaths
- ??

In: 
- Raw EPI

Out:
- Depbliqued, RPI reoriented, slice-time and motion corrected EPI.
- Rigid-body motion parameters.
- Mean EPI image.
- EPI mask.

### Segmentation

Software:
- ANTs Atropos

In:
- Deobliqued, RPI reoriented, skull-stripped T1 (output of anatomical preprocessing)

Out:
- 3 class tissue probability maps (WM, GM, CSF)

### Anatomical Registration

Software:
- ANTs

In:
- Deobliqued, RPI reoriented, bias-corrected, whole-head T1
- MNI template

Out:
- T1 in MNI space
- T1 --> MNI transform

### Functional Registration

Software:
- ANTs

In: 
- Mean EPI image
- Depbliqued, RPI reoriented, slice-time and motion corrected EPI.
- T1 --> MNI transform

Out:
- Mean EPI in MNI space
- EPI in MNI space
