# Report

SCIENCE!
![landmarks on MNI template]
(figures/MNI.png)

## Anatomical preprocessing
Anatomical preprocessing takes the raw T1 weighted image in order to prepare it for future steps in the preprocessing pipeline. The usual steps in anatomical preprocessing include deobliquing the image, reorienting the image to the desired space (in this case MNI/RAS+ space), performing bias reduction, and extracting the brain from the skull. For this project, MNI reorientation was performed first with the help of NIPYPE, followed by a combined bias-reduction/brain extraction with the help of NIPYPE. Finally, the T1 image was deobliqued with the help of the rigid body transformation script in the registration section of the project. All of these steps offered me a useful way to dig deeper into typical preprocessing steps. It highlighted the complexity of neuroimaging as well as the need to fully understand what is under the hood of any functions/programs used in the future.

## Functional preprocessing
Functional preprocessing is the collective term applied to the steps taken from a raw T2 EPI data to prepare it for meaningful analysis with a model (typically, the GLM) and coregistration with an anatomical T1 volume. In SPM terminology, these steps consist of temporal (slice-timing correction) and spatial (realignment / motion correction) preprocessing. Here, we focused on the issues of motion and volumen realignment within a 4D timeseries of volumes. The presented code uses coordinate mapping between volumes to obtain realignment parameters for the 6 rigid body transforms and resample a given volume to the reference, typically the first. This framework output is then compared to SPM's realignment and reslicing functions. Options are available for the use of a first/middle reference volume, one/two realignment passes, and the level of smoothing applied to the data. 

## Segmentation
Segmentation takes the output of anatomical preprocessing and computes the probability that each voxel is white matter, gray matter, and csf. It summarizes those probabilities in three probability maps - one for each tissue class. Here, we tried to generate the probability maps using k-means clustering and the Markov Random Field Expectation-Maximization (MRF-EM) method used in FSL's FAST. Both methods are currently only implemented to run on brain slices.

### k-means clustering
Math/code behind k-means, figures

### MRF-EM
Math/code behind MRF-EM, figures

## Registration
In our code, we write our own methods to find the best full affine transformation to fit match two 3D images, e.g. subject T1 to the MNI template. Four successive searches find the best match (under mutual information) using increasingly more free parameters (translations, 3; plus rotations, 6; plus scales, 9; plus shears, 12). The first search is intialized by matching the center of mass between the two images, and each remaining optmiziation is inialized with the preceding output.

To demonstrate that our image registration procedure is effective, we change the MNI template by a known linear transformation and attempt to recover the initial transformation. The figures below can be generated with `project-red/code/fmri_utils/registration/quality_report.py`.

We transform the MNI template by translating (59, -3, -20) voxels along and rotating (0.2, -0.2, 0.5) radians around the x-, y-, an z-axes.
![change_affine.txt]
(figures/change_affine.txt)

In our registration step, we take the T1 data from each subject and fit it to a common template in MNI space.
We identify specific anatomical landmarks visually on each of the outputs as a way of assessing how effective our registration methods are, and also compare to the known registration package of dipy.


