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

After cloning the repository, put code/fmri_utils onto Python path using setup.py: 

```
pip3 install --user --editable ./code
```

## Running Tests
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

Tests are located in: /code/fmri_utils/tests

## Roadmap 

Picture of how code flows together. [Insert here]

### Data

* We'll give you the data in person or you can download it [here](https://openfmri.org/dataset/ds000030/). It's from the UCLA Consortium for Neuropsychiatric Phenomics LA5c Study. 
* Data files can be found under the data directory. 
    * Path to anatomical data: /data/ds000030/sub-#####/anat
    * Path to functional data: /data/ds000030/sub-#####/func
    * Path to anatomical results: /data/ds000030/sub-#####/anatomical_results

### Validate
* `validate_data.py`: validate data hashes 

### Anatomical preprocessing
* `some code`: de-oblique, skull strip, etc. 

### Functional preprocessing
* `optimize_map_coords.py`: optimizes coordinate mapping
* `volume_realign.py`: realigns volumes in a 4D .nii file

### Segmentation
* `kmeans.py`: does kmeans on the pixel intensity histogram to cluster pixels
* `mrf_em.py`: does Markov Random Field Expectation-Maximization segmentation (used in FSL FAST) 

### Registration
* 


## Discussion *make this sound better*
* limitations: time, our understanding/ability
* take-aways: implementing code helped us understand what happened (better than hand-waving/intuition) & standard packages do something more complicated than the basic idea
* for future: important to inspect analysis stages, not just accept final result

## Authors
* **Chris Muse-Fisher** ([cmusefish](https://github.com/cmusefish))
* **Christine Tseng** ([ctseng12](https://github.com/ctseng12))
* **Jacob Miller** ([jcbmiller94](https://github.com/jcbmiller94))
* **Michael Nagle** ([mpnagle](https://github.com/mpnagle))
* **Zuzanna Balewski** ([zzbalews](https://github.com/zzbalews))
