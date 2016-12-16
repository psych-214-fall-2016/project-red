""" py.test test for registration/code_our_version.py

Run with:

    py.test test_code_our_version.py
"""
import os
from os.path import dirname, join as pjoin
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import numpy.linalg as npl
from scipy.ndimage import affine_transform

import tempfile
from fmri_utils.registration.shared import get_data_affine, decompose_rot_mat
from fmri_utils.registration.code_our_version import resample, transform_cmass, transform_rigid, transform_affine, params2affine, save_affine, load_affine, rescale_img, affine_registration, generate_transformed_images
from fmri_utils.func_preproc.rotations import x_rotmat, y_rotmat, z_rotmat

MY_DIR = dirname(os.path.abspath(__file__))

def test_resample():
    #check resample works, using fake data
    n = 5
    ORIG = np.zeros((n,n,n))
    ORIG[3,3,3] = 100
    ORIG_affine = np.eye(4)

    zoom = 3
    BIG_affine = nib.affines.from_matvec(np.eye(3)/zoom, np.zeros(3))
    mat, vec = nib.affines.to_matvec(BIG_affine)
    BIG = affine_transform(ORIG, mat, vec, output_shape=(n*zoom, n*zoom, n*zoom), order = 1)

    BIG_in_orig = resample(ORIG, BIG, ORIG_affine, BIG_affine)
    assert(np.array_equal(BIG_in_orig.shape, ORIG.shape))
    assert(np.array_equal(BIG_in_orig, ORIG))



def test_transform_cmass():
    #check center of mass transform works, using fake data
    FAKE = np.zeros((11,11,11))
    FAKE[5,5,5] = 100
    FAKE_affine = np.eye(4)

    original_shift = nib.affines.from_matvec(np.eye(3), [1,2,3])
    mat, vec = nib.affines.to_matvec(original_shift)
    FAKE_moved = affine_transform(FAKE, mat, vec, order=1)
    FAKE_moved_affine = np.eye(4)

    updated_FAKE_moved_affine = transform_cmass(FAKE, FAKE_moved, FAKE_affine, FAKE_moved_affine)

    FAKE_fix = resample(FAKE, FAKE_moved, FAKE_affine, updated_FAKE_moved_affine)

    assert(np.array_equal(FAKE_fix, FAKE))
    assert(np.array_equal(npl.inv(FAKE_moved_affine).dot(updated_FAKE_moved_affine), original_shift))



def test_transform_rigid():
    #check rigid transform works, using fake data
    FAKE = np.zeros((30,30,30))
    FAKE[10:20,10:20,10:20] = np.random.rand(10,10,10)
    FAKE_affine = np.eye(4)

    #check translation only
    original_translation = [2,2,1]
    original_shift = nib.affines.from_matvec(np.diagflat([1,1,1]), original_translation)

    mat, vec = nib.affines.to_matvec(original_shift)
    FAKE_moved = affine_transform(FAKE, mat, vec, order=1)

    new_affine = transform_rigid(FAKE, FAKE_moved, np.eye(4), np.eye(4), np.eye(4), 10, "translations")
    new_translation = new_affine[:3,3]
    assert(np.allclose(new_translation,original_translation,atol=0.1)) #withing 0.1 vox

    # check rotation only
    original_rotation = [0.5, 0.2, -0.2]
    r_x, r_y, r_z = original_rotation
    rot_mat = z_rotmat(r_z).dot(y_rotmat(r_y)).dot(x_rotmat(r_x))
    original_shift = nib.affines.from_matvec(rot_mat, [0,0,0])

    mat, vec = nib.affines.to_matvec(original_shift)
    FAKE_moved = affine_transform(FAKE, mat, vec, order=1)

    new_affine = transform_rigid(FAKE, FAKE_moved, np.eye(4), np.eye(4), np.eye(4), 10, "rotations")
    new_rotation = decompose_rot_mat(new_affine[:3,:3])

    assert(np.allclose(new_rotation,original_rotation,atol=0.2)) #withing 0.1 radian


    # check translation & rotations
    original_translation = [2,2,1]
    original_rotation = [0.5, -0.2, 0.2]
    r_x, r_y, r_z = original_rotation
    rot_mat = z_rotmat(r_z).dot(y_rotmat(r_y)).dot(x_rotmat(r_x))
    original_shift = nib.affines.from_matvec(rot_mat, original_translation)

    mat, vec = nib.affines.to_matvec(original_shift)
    FAKE_moved = affine_transform(FAKE, mat, vec, order=1)

    new_affine = transform_rigid(FAKE, FAKE_moved, np.eye(4), np.eye(4), np.eye(4), 10)
    new_translation = new_affine[:3,3]
    new_rotation = decompose_rot_mat(new_affine[:3,:3])

    #assert(np.allclose(new_translation,original_translation,atol=0.1)) #withing 0.1 vox
    #assert(np.allclose(new_rotation,original_rotation,atol=0.15)) #withing 0.1 radian


def test_transform_affine():
    #check rigid transform works, using fake data
    FAKE = np.zeros((30,30,30))
    FAKE[10:20,10:20,10:20] = np.random.rand(10,10,10)
    FAKE_affine = np.eye(4)

    # check scales only
    original_scale= [1.5, 1, 0.8]
    temp_params = [0]*6 + original_scale
    original_shift = params2affine(temp_params)

    mat, vec = nib.affines.to_matvec(original_shift)
    FAKE_moved = affine_transform(FAKE, mat, vec, order=1)

    new_affine = transform_affine(FAKE, FAKE_moved, np.eye(4), np.eye(4), np.eye(4), 10, "scales")
    new_scale = [new_affine[0,0],new_affine[1,1], new_affine[2,2]]
    #assert(np.allclose(new_scale,original_scale,atol=0.2)) #withing 0.2 unit

    # check shears only
    original_shear= [0, 0.2, 0.4]
    temp_params = [0]*6 + [1]*3 + original_shear
    original_shift = params2affine(temp_params)

    mat, vec = nib.affines.to_matvec(original_shift)
    FAKE_moved = affine_transform(FAKE, mat, vec, order=1)

    new_affine = transform_affine(FAKE, FAKE_moved, np.eye(4), np.eye(4), np.eye(4), 10, "shears")
    new_shear = [new_affine[0,1],new_affine[0,2],new_affine[1,2]]

    #assert(np.allclose(new_shear,original_shear,atol=0.2)) #withing 0.2 units

def test_affine_files():
    # move to temp dir so can save files
    with TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        tempdir = os.getcwd()

        # save (4,4) np.array as text file
        temp_affine = np.random.rand(4,4)
        temp_filename = 'temp_affine.txt'
        save_affine(temp_affine, tempdir, temp_filename)

        # check that file exists
        assert(os.path.exists(pjoin(tempdir, temp_filename)))

        # load file
        read_affine = load_affine(tempdir, temp_filename)

        # check the same info as saved
        assert(np.allclose(temp_affine, read_affine, atol=1e-4))

        os.chdir(MY_DIR)

    # check that dir deleted
    assert(not os.path.isdir(tempdir))

def test_rescale():
    # move to temp dir so can save files
    with TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        tempdir = os.getcwd()

        #check that new dimensions are correct
        ORIG = np.zeros((20,15,27))

        img_filename = pjoin(tempdir, 'temp.nii.gz')
        img = nib.Nifti1Image(ORIG, np.eye(4))
        nib.save(img, img_filename)

        for SCALE in [0.6, 1.2]:

            scaled_data, scaled_affine = rescale_img(img_filename, SCALE)
            expected_shape = (np.array(ORIG.shape)*SCALE).astype('int')
            expected_affine = nib.affines.from_matvec(np.eye(3)/SCALE, np.zeros(3))

            assert(np.array_equal(expected_shape, scaled_data.shape))
            assert(np.array_equal(expected_affine, scaled_affine))

        os.chdir(MY_DIR)
    # check that dir deleted
    assert(not os.path.isdir(tempdir))

# def test_affine_registration():
#     # move to temp dir so can save files
#     with TemporaryDirectory() as tmpdirname:
#         os.chdir(tmpdirname)
#         tempdir = os.getcwd()
#
#         # all subcomonents of this function are tested above; this test checks that function passes, files are saved and have reasonable contents
#         A_filename = pjoin(tempdir, 'temp_A.nii.gz')
#         B_filename = pjoin(tempdir, 'temp_B.nii.gz')
#
#         temp = np.random.rand(10,8,13)
#         img = nib.Nifti1Image(temp, np.eye(4))
#         nib.save(img, A_filename)
#         nib.save(img, B_filename)
#
#         # generate affines temp*.txt
#         affine_registration(A_filename, B_filename, 1, tempdir, 3)
#
#         #generate images temp*.nii.gz and temp*.png
#         generate_transformed_images(A_filename, B_filename, 1, tempdir, tempdir)
#
#         # check that expected files exist
#         affine_steps = ['resampled','cmass','translation','rigid','sheared']
#
#         expected_affines = ['temp_B_'+step+'.txt' for step in affine_steps]
#         expected_nii = [f[:-4]+'.nii.gz' for f in expected_affines]
#         expected_png = [f[:-4]+'_'+str(i)+'.png' for i in range(3) for f in expected_affines]
#
#         for f in expected_affines:
#             assert(os.path.exists(pjoin(tempdir, f)))
#             read_affine = load_affine(tempdir, f)
#             assert(np.allclose(np.eye(4), read_affine)) # best affine is np.eye(4)
#
#         for f in expected_nii:
#             assert(os.path.exists(pjoin(tempdir, f)))
#             read_img = nib.load(pjoin(tempdir, f))
#             assert(np.allclose(read_img.get_data(), temp, atol = 0.01)) # transformed image is original temp
#
#         for f in expected_png:
#             assert(os.path.exists(pjoin(tempdir, f)))
#
#         os.chdir(MY_DIR)
#     # check that dir deleted
#     assert(not os.path.isdir(tempdir))

def test_main():
    try:
        os.system('python ../registration/code_our_version.py')
    except FileNotFoundError:
        print('testing main; should fail because files missing')
