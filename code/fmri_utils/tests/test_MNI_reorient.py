from os.path import dirname , join as pjoin
import nibabel as nib
from MNI_reorient import MNI_reorient

# name of the file with the incorrect orientation
ROOTDIR = dirname(__file__)
ROOT_DATA_DIR = pjoin(ROOTDIR, '../../../data')
bad_orientation_file = pjoin(ROOT_DATA_DIR, 'MNI_template/T1w_RPI.nii.gz')
# name of the file to be created after the MNI reorientation
out_file_name = pjoin(ROOT_DATA_DIR, 'MNI_template/T1w_RAS.nii.gz')

# a string with the desired orientation code (this is the format that
# nib.aff2axcodes() returns)

MNI_axis_codes = ('R', 'A', 'S')
def test_MNI_reorient():
    # load up the image with the incorrect orientation
    bad_img = nib.load(bad_orientation_file)
    # get the affine information from the image
    bad_affine = bad_img.affine
    # get the orientation code from the affine information
    bad_orientation = nib.aff2axcodes(bad_affine)

    # run the reorientation on the file with the bad orientation to create
    # a file with the correct orientation
    MNI_reorient(bad_orientation_file, out_file_name)

    # load up the image data of the out file created in the previous step
    out_file_img = nib.load(out_file_name)
    # get the affine information from the out file image
    out_file_affine = out_file_img.affine
    # get the orientation code from the affine information
    out_file_orientation = nib.aff2axcodes(out_file_affine)

    # assert that the original image with the incorrect orientation actually does
    # have a different orientation than what is desired
    assert(bad_orientation != MNI_axis_codes)
    # assert that the newly created out file has had its orientation swapped
    # to the desired orientation
    assert(out_file_orientation == MNI_axis_codes)
