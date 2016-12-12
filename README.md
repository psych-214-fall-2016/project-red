# Preprocessing in the 21st Century: A Musical Extravaganza (Project Red)

Fall 2016 final project for PSY 214. In order to avoid using preprocessing without understanding what is happening, we  investigated a few main steps from a standard preprocessing pipeline. We attempted to implement anatomical preprocessing, functional preprocessing, segmentation, and registration. 

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

### Installing

After cloning the repository, put `code/fmri_utils` onto your Python path using setup.py: 

```
pip3 install --user --editable ./code
```

### Getting the Data

The data is from the UCLA Consortium for Neuropsychiatric Phenomics LA5c Study. We can give it to you in person or you can download it [here](https://openfmri.org/dataset/ds000030/). 

Put data files in the data directory so that the paths are in the following format:

* Path to anatomical data: `/data/ds000030/sub-#####/anat`
* Path to functional data: `/data/ds000030/sub-#####/func`

Scripts will generate outputs in the following format:

* Path to anatomical preprocessing results: `/data/ds000030/sub-#####/anatomical_results`
* Path to functional preprocessing results: `/data/ds000030/sub-#####/functional_results`
* Path to segmentation results: `/data/ds000030/sub-#####/segmentation_results`
* Path to anatomical registration results: `/data/ds000030/sub-#####/registration_results`


## Testing/Validation

### Validate data
Check the integrity of your data by running

```
python3 /code/fmri_utils/data_hashes.py data
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

## Roadmap 

Picture of how code flows together. [Insert here]

### Anatomical preprocessing
* `some code`: de-oblique, skull strip, etc. 

### Functional preprocessing
* `optimize_map_coords.py`: optimizes coordinate mapping
* `volume_realign.py`: realigns volumes in a 4D .nii file

### Segmentation
Segmentation takes the output of anatomical preprocessing and computes the probability that each voxel is white matter, gray matter, and csf. It summarizes those probabilities in three probability maps - one for each tissue class. Here, we tried to generate the probability maps using k-means clustering and the Markov Random Field Expectation-Maximization (MRF-EM) method used in FSL's FAST. Both methods are currently only implemented to run on brain slices. 

* `kmeans.py`: does k-means on the pixel intensity histogram (< 30 s)
* `mrf_em.py`: does MRF-EM segmentation (used in FSL FAST) (~ 6 min for a 20x20 pixel slice of brain)

### Registration


## Discussion
Although we'd hoped to implement each step fully, most were implemented as simpler versions of the corresponding steps in standard preprocessing pipelines. We were mainly limited by time and prior understanding/experience with coding and preprocessing. However, writing and testing code from scratch gave us a much better understanding of what the pipelines do,  and it underscored the complexity of these steps beyond the basic hand-wavy/intuitive ideas. Our main takeaway is that it's important to inspect analysis stages and not just accept final results. 

## Authors
* **Chris Muse-Fisher** ([cmusefish](https://github.com/cmusefish))
* **Christine Tseng** ([ctseng12](https://github.com/ctseng12))
* **Jacob Miller** ([jcbmiller94](https://github.com/jcbmiller94))
* **Michael Nagle** ([mpnagle](https://github.com/mpnagle))
* **Zuzanna Balewski** ([zzbalews](https://github.com/zzbalews))
* **Dan Lurie (mentor)** ([danlurie](https://github.com/danlurie))
