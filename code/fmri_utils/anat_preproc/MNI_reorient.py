from nipype.interfaces import fsl as fsl


# this function swaps the dimensions of the file by doing 90, 180, 0r 270 degree
# rotations. The final product is in the RAS+ orientation (Right, Anterior, and Superior parts of the brain
# are the positive values).
def MNI_reorient(structural_nifti_file , out_file):
    MNI = fsl.Reorient2Std()
    MNI.inputs.in_file = structural_nifti_file
    MNI.inputs.out_file = out_file
    MNI_results = MNI.run()
    return MNI_results



#MNI_reorient('/Users/despolab/CMF_Files/data/ds000030/sub-10171/anat/sub-10171_T1w.nii.gz', '/Users/despolab/CMF_Files/data/ds000030/sub-10171/anat/sub-10171_T1w_MNI.nii.gz')
