from nipype.interfaces import fsl as fsl
# The main purpose of having this function is to swap the orientation in order to
# test that the MNI_reorient.py function works.

# I called this function on a random file to swap the dimensions into the RPI+ orientation.
# This is the bad_orientation file in the test_MNI_reorient function.

# The way the fsl.SwapDimensions works is by looking at the affine information within
# the HDR to determine which orientation it is already in. It then changes the +/- signs on
# certain parts of the affine matrix to put it into the desired orientation. This function
# applies 90, 180, or 270 degree rotations to the data without altering the relative location
# of the data.

def RPI_reorient(structural_nifti_file, out_file):
    RPI = fsl.SwapDimensions()
    RPI.inputs.in_file = structural_nifti_file
    # the desired location of the output file
    RPI.inputs.out_file = out_file
    # input the desired dimensions. If it is a Nifti file, 'RL'/'LR', 'AP'/'PA',
    # and 'SI'/'IS' can be entered. If not, x/-x, y/-y, and z/-z need to be entered.
    RPI.inputs.new_dims = ('LR' , 'AP' , 'SI')
    RPI_results = RPI.run()
    return RPI_results
