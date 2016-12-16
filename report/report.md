# Report

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
In our code, we write our own methods to find the best full affine transformation to fit match two 3D images, e.g. subject T1 to the MNI template. Four successive searches find the best match (under mutual information) using increasingly more free parameters (translations, 3; plus rotations, 6; plus scales, 9; plus shears, 12). The first search is intialized by matching the center of mass between the two images, and each remaining optmiziation is inialized with the preceding output. We are using linear interpolation whenever resampling is required.

We are using skull-stripped images because the outside shape contributes meaningfully to the optimization procedure, and we are aiming to match brains not head/neck shapes. The examples below register the MNI template with itself or with individual subject T1 images, but the procedure is agnostic to the types of images being compared and could also be used to register individual T1 and sample T2\* volumes.


### Does the registration procedure work?
To demonstrate that our image registration procedure is effective, we change the MNI template by a known linear transformation and attempt to recover the initial transformation. The figures below can be generated with `project-red/code/fmri_utils/registration/quality_report.py` (~1 hr to run).

We transform the MNI template by translating (59, -3, -20) voxels along and rotating (0.2, -0.2, 0.5) radians around the x-, y-, an z-axes. We will call the original MNI template the "static" image, and the new transformed MNI template the "moving" image.

This set of figures shows cross sections of the static image on the left, the same cross sections of moving image on the right, and the overlap in the middle (green = left, red = right, yellow = overlap). 

![resampled_0]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_resampled_0.png)

![resampled_1]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_resampled_1.png)

![resampled_2]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_resampled_2.png)

We start the registration process by translating the moving image to match the center of mass with the static image.

![cmass_0]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_cmass_0.png)

![cmass_1]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_cmass_1.png)

![cmass_2]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_cmass_2.png)

The first optimization finds the best translation parameters (3) to minimize negative mutual information between the static and moving images, intialized with the above center of mass transform.

![translation_0]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_translation_0.png)

![translation_1]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_translation_1.png)

![translation_2]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_translation_2.png)

The second optimization find the best rigid transform (translation and rotation) parameters (6), initalized with the best parameters from the previous step.

![rigid_0]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_rigid_0.png)

![rigid_1]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_rigid_1.png)

![rigid_2]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_rigid_2.png)

The third optimization find the best translation, rotation, and shearing parameters (9); the fourth optimization finds the best translation, rotation, shearing, and scaling parameters (12). Both are initalized with the best parameters from the previous step. Since the results are so similar in this case (no scaling or shearing was applied in the initial transform), we will show the results of the final full affine transformation.

![MNI_resampled_0]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_sheared_0.png)

![MNI_resampled_1]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_sheared_1.png)

![MNI_resampled_2]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_sheared_2.png)


At each step the best parameters are minizing a cost function, in this case the negative mutual information between the two images. This plot shows the negative mutual information between the static image and:
* itself (red): ideal minimum
* inverse transform of moving image (green): actual minimum-- some information is lost due to resampling
* transformed moving image
* center of mass transform of moving image
* translation transform of moving image
* rigid (translation & rotation) transform of moving image
* full affine (translation, rotation, scaling, & shearing) transform of moving image
![neg_MI]
(figures/mni_icbm152_t1_tal_nlin_asym_09a_brain_changed_MI.png)

From the overlay illustrations and negative mutual information plot, we are satisfied that our registration is successfully recovering the inital transform (given that some information is necessarily lost in the process of resampling). We feel confident enough in our process to proceed to a scientifically more interesting question:

### How well does the registration procedure work for aligning individual subject T1s to the MNI template?
Registering individual subject T1s to the MNI template is a much harder problem because, in addition to being translated and rotated, individual brains have different overall shapes, patterns of sulci and gyri, and may have a different distribution of intensity values. 

We take the T1 images from 7 subjects and register them to the MNI template using the same procedure described above. We then identify specific anatomical landmarks manually on each of the outputs to qualitatively asses how effective our registration methods are. The figures below can be generated with `project-red/code/fmri_utils/registration/registration_report.py`. Since the fitting procedure takes ~1 hr for each subject, we have saved the best affine transforms from each registration step; to rerun this registration uncomment line ##. 

We'll look at one sample subject (sub-10159) to illustrate what the registration procedure starts and ends with for a real T1 to MNI match. This is where the registration starts (matching centers of mass):

![cmass_0]
(figures/sub-10159_T1w_brain_cmass_backup_0.png)

![cmass_1]
(figures/sub-10159_T1w_brain_cmass_backup_1.png)

![cmass_2]
(figures/sub-10159_T1w_brain_cmass_backup_2.png)

And this is where it ends (full affine transform):

![sheared_0]
(figures/sub-10159_T1w_brain_sheared_backup_0.png)

![sheared_1]
(figures/sub-10159_T1w_brain_sheared_backup_1.png)

![sheared_2]
(figures/sub-10159_T1w_brain_sheared_backup_2.png)

Let's look at the saggital plane for the remaining 6 subjects:

![sheared_2]
(figures/sub-10171_T1w_brain_sheared_backup_2.png)

![sheared_2]
(figures/sub-10189_T1w_brain_sheared_backup_2.png)

![sheared_2]
(figures/sub-10193_T1w_brain_sheared_backup_2.png)

![sheared_2]
(figures/sub-10206_T1w_brain_sheared_backup_2.png)

![sheared_2]
(figures/sub-10217_T1w_brain_sheared_backup_2.png)

![sheared_2]
(figures/sub-10225_T1w_brain_sheared_backup_2.png)

We can say that the transformed T1 brains look similar to the MNI template, but it's hard to evaluate the success of the registration from this kind of visual inspection. We decided to manually mark a few prominent landmarks on these registered brain and compare their locations to the expected coordinates on the MNI template. Our labeling procedure was: 
* locate the anterior commissure (x=0, y=0, z=0mm in MNI) in the sagital plane for each subject
* on this z-place, get (x,y) coordinates for the right anterior and posterior insula, left and right ventricle peaks, and start of corpos callosum on the midline.

The following plots show the full affine transformed T1 for each subject; saggital view on the left and axial view on the right; subject coordinates in green and MNI coordinates in red.

![landmarks]
(figures/sub-10159.png)

![landmarks]
(figures/sub-10171.png)

![landmarks]
(figures/sub-10189.png)

![landmarks]
(figures/sub-10193.png)

![landmarks]
(figures/sub-10206.png)

![landmarks]
(figures/sub-10217.png)

![landmarks]
(figures/sub-10225.png)

Conclusion?
### How do our results compare to a similar registration procedure in the dipy package?


