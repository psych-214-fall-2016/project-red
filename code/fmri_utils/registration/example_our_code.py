from os.path import dirname, join as pjoin
import sys

import nibabel as nib
import numpy as np
import numpy.linalg as npl

import dipy
from dipy.viz import regtools
import numpy.linalg as npl
from scipy.ndimage import affine_transform

from fmri_utils.registration.shared import get_data_affine, decompose_rot_mat
from fmri_utils.registration.code_our_version import resample, transform_cmass, transform_rigid, transform_affine

def do_example(mni_filename, SUBJ, SCALE, from_saved, input_dir, output_dir):
    print('Working on: '+SUBJ)

    if from_saved:
        print('... skipping optimizations; using existing *_backup.txt files\n')
    else:
        print('... running all optimization steps\n')


    # load data
    print('MNI template: '+mni_filename)
    if from_saved:
        subj_filename = pjoin(input_dir,SUBJ+'_T1w_brain.nii.gz')
    else:
        subj_filename = pjoin(input_dir,SUBJ+'_T1w_skull_stripped.nii.gz')
    print('SUBJ file: '+subj_filename)

    static, moving, static_affine, moving_affine = get_rescaled_data(mni_filename, subj_filename, SCALE)

    # resample into template space
    print('--- working on *resampled*')
    if from_saved:
        resample_affine = load_affine(pjoin(input_dir,SUBJ+'_T1w_brain_resampled_backup.txt'))
    else:
        resample_affine = np.eye(4)
    updated_moving_affine = resample_affine.dot(moving_affine)

    resampled = resample(static, moving, static_affine, updated_moving_affine)
    save_outputs(SUBJ, resampled, static_affine, resample_affine, 'resampled', output_dir, static)

    # center of mass transform
    print('--- working on *cmass*')
    if from_saved:
        cmass_affine = load_affine(pjoin(input_dir,SUBJ+'_T1w_brain_cmass_backup.txt'))
    else:
        cmass_affine = transform_cmass(static, moving, static_affine, moving_affine)
    updated_moving_affine = cmass_affine.dot(moving_affine)

    cmass = resample(static, moving, static_affine, updated_moving_affine)
    save_outputs(SUBJ, cmass, static_affine, cmass_affine, 'cmass', output_dir, static)


    ## rigid: translation only
    print('--- working on *translation*')
    if from_saved:
        translation_affine = load_affine(pjoin(input_dir,SUBJ+'_T1w_brain_translation_backup.txt'))
    else:
        translation_affine = transform_rigid(static, moving, static_affine, moving_affine, cmass_affine, 10, "translations")
    updated_moving_affine = translation_affine.dot(moving_affine)

    translation = resample(static, moving, static_affine, updated_moving_affine)
    save_outputs(SUBJ, translation, static_affine, translation_affine, 'translation', output_dir, static)

    ## rigid: translation and rotation
    print('--- working on *rigid*')
    if from_saved:
        rigid_affine = load_affine(pjoin(input_dir,SUBJ+'_T1w_brain_rigid_backup.txt'))
    else:
        rigid_affine = transform_rigid(static, moving, static_affine, moving_affine, translation_affine, 10, "all")
    updated_moving_affine = rigid_affine.dot(moving_affine)

    rigid = resample(static, moving, static_affine, updated_moving_affine)
    save_outputs(SUBJ, rigid, static_affine, rigid_affine, 'rigid', output_dir, static)


    ## affine: translation, rotation, scaling, and shearing
    print('--- working on *sheared*')
    if from_saved:
        shearing_affine = load_affine(pjoin(input_dir,SUBJ+'_T1w_brain_sheared_backup.txt'))
    else:
        shearing_affine = transform_affine(static, moving, static_affine, moving_affine, rigid_affine, 10, "all")
    updated_moving_affine = shearing_affine.dot(moving_affine)

    sheared = resample(static, moving, static_affine, updated_moving_affine)
    save_outputs(SUBJ, sheared, static_affine, shearing_affine, 'sheared', output_dir, static)


def save_outputs(subject, data, static_affine, transform_affine, section, output_dir, static):
    # save slice overlays
    regtools.overlay_slices(static, data, None, 0, "Static", "Moving", output_dir+subject+"_"+section+"_0.png")
    regtools.overlay_slices(static, data, None, 1, "Static", "Moving", output_dir+subject+"_"+section+"_1.png")
    regtools.overlay_slices(static, data, None, 2, "Static", "Moving", output_dir+subject+"_"+section+"_2.png")

    # save resampled *.nii.gz images
    img = nib.Nifti1Image(data, static_affine)
    nib.save(img, output_dir+subject+'_T1w_brain_'+section+'.nii.gz')

    # save affines
    f = open(output_dir+subject+'_T1w_brain_'+section+'.txt', 'w')
    f.write(str(transform_affine))
    f.close()

def load_affine(file):
    affine_str = [i.replace('[','').replace(']','').split() for i in open(file,'r')]
    affine_float = [[float(y) for y in row] for row in affine_str]
    affine = np.array(affine_float)
    return affine

def get_rescaled_data(static_filename, moving_filename, SCALE):
    # load data
    template = nib.load(static_filename)
    subject = nib.load(moving_filename)

    static = template.get_data()
    static_affine = template.affine

    moving = subject.get_data()
    moving_affine = subject.affine

    # set downsample
    SCALE_affine = nib.affines.from_matvec(np.diagflat([1/SCALE]*3), np.zeros(3))

    static_scaled_shape = (np.array(static.shape)*SCALE).astype('int')
    moving_scaled_shape = (np.array(moving.shape)*SCALE).astype('int')

    static_scaled_affine = static_affine.dot(SCALE_affine)
    moving_scaled_affine = moving_affine.dot(SCALE_affine)

    # resample
    static_scaled = resample(np.zeros(static_scaled_shape), static, static_scaled_affine, static_affine)
    moving_scaled = resample(np.zeros(moving_scaled_shape), moving, moving_scaled_affine, moving_affine)

    return static_scaled, moving_scaled, static_scaled_affine, moving_scaled_affine


def main():
    errormsg = "Need to state if using saved results or rerunning optimizations. \
    \nOptions:\n\
    $ python3 example_our_code.py saved\n\
    $ python3 example_our_code.py rerun"

    if len(sys.argv) < 2:
        raise RuntimeError(errormsg)

    if sys.argv[1]=='saved':
        from_saved = True
    elif sys.argv[1]=='rerun':
        from_saved = False
    else:
        raise RuntimeError(errormsg)


    subjects = ['sub-10159', 'sub-10171', 'sub-10189', 'sub-10193', 'sub-10206', 'sub-10217', 'sub-10225']
    SCALE = 1

    for SUBJ in subjects:
        if from_saved:
            mni_filename= '../../../data/registration_example_files/mni_icbm152_t1_tal_nlin_asym_09a_brain.nii.gz'
            input_dir = pjoin('../../../data/registration_example_files')
        else:
            mni_filename= '../../../data/MNI_template/mni_icbm152_t1_tal_nlin_asym_09a_skull_stripped.nii.gz'
            input_dir = pjoin('../../../data/ds000030',SUBJ,'anat')

        output_dir = pjoin('../../../data/ds000030',SUBJ,'registration_results')

        do_example(mni_filename, SUBJ, SCALE, from_saved, input_dir, output_dir) #reusing saved outputs


if __name__ == '__main__':
    main()
