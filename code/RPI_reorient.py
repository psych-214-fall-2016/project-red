#Re-orienting the Image into Right-to-Left Posterior-to-Anterior Inferior-to-Superior (RPI) orientation
from nipype.interfaces import fsl as fsl

def RPI_reorient(structural_nifti_file, out_file):
    RPI = fsl.SwapDimensions()
    RPI.inputs.in_file = structural_nifti_file
    RPI.inputs.out_file = out_file
    RPI.inputs.new_dims = ('LR' , 'AP' , 'SI')
    RPI_results = RPI.run()
    return RPI_results

RPI_reorient('/Users/despolab/CMF_Files/data/ds000030/sub-10171/anat/sub-10171_T1w.nii.gz', '/Users/despolab/CMF_Files/data/ds000030/sub-10171/anat/sub-10171_T1w_RPI.nii.gz')
