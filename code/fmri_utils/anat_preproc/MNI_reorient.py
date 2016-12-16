from nipype.interfaces import fsl as fsl



def MNI_reorient(structural_nifti_file , out_file):
    """This function swaps the dimensions of the file by doing 90, 180, 0r 270 degree
    rotations. The final product is in the RAS+ orientation (Right, Anterior, and Superior parts of the brain
    are the positive values).

    - in order to swap the dimensions, fsl.Reorient2Std calls fsl.SwapDimensions. In other words,
    it is a specified version of Reorient2Std. For more info on Reorient2Std, look at RPI_reorient.py


    input
    ------
    structural_nifti_file: nifti/nifti.gz file
    The original T1 weighted anatomical file

    output
    ------
    out_file: nifti.gz file
    an MNI reoriented image in the RAS+ orientation

    """


    MNI = fsl.Reorient2Std()
    MNI.inputs.in_file = structural_nifti_file
    MNI.inputs.out_file = out_file
    MNI_results = MNI.run()
    return MNI_results
