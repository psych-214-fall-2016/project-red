# project-template

Fall 2016 final project.

This file is in [Markdown
format](http://daringfireball.net/projects/markdown), and should render nicely
on the Github front page for this repository.

## Install

To install the necessary code:

    # Install required packages
    pip3 install --user -r requirements.txt
    # Put code/fmri_utils onto Python path using setup.py
    pip3 install --user --editable ./code

To run tests:

* install `pytest` with ``pip3 install --user pytest``;
* run tests with:

    py.test fmri_utils

## Test

Install pytest:

    pip3 install --user pytest

Run the tests:

    py.test code

## Roadmap 
Picture of how code flows together. [Insert here]
### Data
* We'll give you the data in person. It's from the Internet [insert link]
### Validate
* `validate_data.py`: validate data hashes 
### Tests
* Run them yourself.
### Anatomical preprocessing
* `some code`: de-oblique, skull strip, etc. 
### Functional preprocessing
* `optimize_map_coords.py`: optimizes coordinate mapping
* `volume_realign.py`: realigns volumes in a 4D .nii file
### Segmentation
* `kmeans.py`: does kmeans on pixel intensity histogram to cluster pixels
* `mrf_em.py`: does MRF EM 
### Registration
* 


