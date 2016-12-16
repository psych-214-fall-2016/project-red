# Preprocessing in the 21st Century: [A Musical Extravaganza](https://www.youtube.com/watch?v=hm7_elsseuU) (Project Red)

Fall 2016 final project for PSY 214.
Before fMRI data can be meaningully analyzed, a series of steps to better prepare the data for analysis are taken: 'preprocessing.' All fMRI analysis packages implement a generally common framework for these steps, although the exact alogirthms applied and the decisions made are often not available, interpretabe, or understood. Rather than apply these steps without understanding what is happening in each one, we  investigated a few main steps from the 'standard' preprocessing pipeline. We attempted to implement anatomical preprocessing, functional preprocessing, segmentation, and registration mostly from scratch in Python, and compare our funcitonality in mechanism and results to that of available fMRI packages.

This file is in [Markdown
format](http://daringfireball.net/projects/markdown), and should render nicely
on the Github front page for this repository.

## Getting started

This will get a copy of the project up and running on your own machine!

### Prerequisites

Run the following to install required packages:

```
pip3 install --user -r requirements.txt
```
Additionally, FSL is needed for anatomical preprocessing. Instructions for the download can be found [here](https://fsl.fmrib.ox.ac.uk/fsldownloads/fsldownloadmain.html).

### Installing

After cloning the repository, put `code/fmri_utils` onto your Python path using setup.py:

```
pip3 install --user --editable ./code
```

### Getting the Data

The data is from the UCLA Consortium for Neuropsychiatric Phenomics LA5c Study. We can give it to you in person or you can download it [here](https://openfmri.org/dataset/ds000030/).

Put data files in the data directory so that the necessary files are found in these paths:

* Path to anatomical data: `/data/ds000030/sub-#####/anat/sub-#####_T1w.nii.gz`
* Path to functional data: `/data/ds000030/sub-#####/func/sub-#####_task-rest_bold.nii.gz`

You will also need the MNI template file. You can download it [here](https://bic-berkeley.github.io/psych-214-fall-2016/_downloads/mni_icbm152_t1_tal_nlin_asym_09a.nii).

Put the MNI template in the data directory: `/data/mni_icbm152_t1_tal_nlin_asym_09a.nii`.

Scripts in this package will create these directories (if they do not already exist) for generated files:

* Path to anatomical preprocessing results: `/data/ds000030/sub-#####/anatomical_results`
* Path to functional preprocessing results: `/data/ds000030/sub-#####/functional_results`
* Path to segmentation results: `/data/ds000030/sub-#####/segmentation_results`
* Path to anatomical registration results: `/data/ds000030/sub-#####/registration_results`


## Testing/Validation

### Validate data
Check the integrity of your data by running

```
python3 /code/fmri_utils/check_data_org.py
```

This compares generated hashes for the data you downloaded with hashes in `/data/data_hashes.txt`.

### Running tests
To run tests, you should have pytest installed. Install pytest by running

 ```
 pip3 install --user pytest
 ```

Run all the tests by running

```
py.test fmri_utils
```

and a specific test by running

```
py.test code
```

where ```code``` is the name of the test you want to run.    

Tests are located in: `/code/fmri_utils/tests`

## Making figures
Make figures for each section by running the following:
* `make anatomical_preproc`
* `make functional_preproc`
* `make segmentation`
* `make registration`

## Roadmap

Picture of how code flows together. [Insert here]

### Anatomical preprocessing
* `run_anat_preprocess.py`: master file that calls the rest of the anatomical preproc steps
* `MNI_reorient.py`: orient the image into RAS+ orientation
* `skull_strip.py`: performs bias reduction and removes the skull

### Functional preprocessing
* `optimize_map_coords.py`: optimizes coordinate mapping
* `volume_realign.py`: realigns volumes in a 4D .nii file

### Segmentation
* `kmeans.py`: does k-means on the pixel intensity histogram (< 30 s)
* `mrf_em.py`: does MRF-EM segmentation (used in FSL FAST) (~ 6 min for a 20x20 pixel slice of brain)

### Registration
* `code_from_dipy.py`: a wrapper around dipy registration functions
* `example_from_dipy.py`: adapted example from [here](http://nipy.org/dipy/examples_built/affine_registration_3d.html#example-affine-registration-3d).
* `code_our_version.py`: our registration functions which optimize by center of mass, translation, rotations, and shear/scaling.
* `shared.py`: general useful functions
* `quality_report.py`: code which uses the regstration method in `code_our_version.py` on template with known transform to show it works.
* `registration_report.py`: code which uses the regstration method in `code_our_version.py` on seven specific subjects to show its results.

## Discussion
Although we'd hoped to implement each step fully, most were implemented as simpler versions of the corresponding steps in standard preprocessing pipelines. We were mainly limited by time and prior understanding/experience with coding and preprocessing. However, writing and testing code from scratch gave us a much better understanding of what the pipelines do,  and it underscored the complexity of these steps beyond the basic hand-wavy/intuitive ideas. Our main takeaway is that it's important to inspect analysis stages and not just accept final results.

## Authors
* **Chris Muse-Fisher** ([cmusefish](https://github.com/cmusefish))
* **Christine Tseng** ([ctseng12](https://github.com/ctseng12))
* **Jacob Miller** ([jcbmiller94](https://github.com/jcbmiller94))
* **Michael Nagle** ([mpnagle](https://github.com/mpnagle))
* **Zuzanna Balewski** ([zzbalews](https://github.com/zzbalews))
* **Dan Lurie (mentor)** ([danlurie](https://github.com/danlurie))
