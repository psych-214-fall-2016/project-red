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
* for our specific data, the files can be found under the data directory. 
* the path to anatomical data is /data/ds000030/sub-#####/anat
* the path to functional data is /data/ds000030/sub-#####/func
* the path to anatomical results is /data/ds000030/sub-#####/anatomical_results
* 

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


## Discussion *make this sound better*
* motivation: in order to avoid using preprocessing without understanding what is happening, we want to investigate a few (main? most common?) steps from a standard preprocessing pipeline
* limitations: time, our understanding/ability
* take-aways: implementing code helped us understand what happened (better than hand-waving/intuition) & standard packages do something more complicated than the basic idea
* for future: important to inspect analysis stages, not just accept final result


