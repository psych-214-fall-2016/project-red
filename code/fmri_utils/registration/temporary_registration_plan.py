# organizational file for registration module plan

"""
from the dipy tutorial:

0. resample moving into static space; show overlay slices
1. center of mass alignment for static;
2. rigid transform (using com affine as seed)
3. affine transform (using rigid affine as seed)

—> all use 3 level pyramid for each optimization

our plan:

- organize code from class into usable funcs; similar outputs as dipy steps above
- write tests
    1. are we doing what we think we’re doing? use toy ex with known changes
    2. how do our results compare to dipy
- add in pyramid strategy and compare accuracy/efficiency; compare to dipy again

** write all code using data for subject10159;
"""
#### FUNCTIONS TO WRITE
## shared.py

#load data into python; define `static` and `moving` [Zuzanna]


## code_from_dipy.py [Michael]

#resample (dipy); from static and moving, produce new affine

#center of mass transform (dipy); from static and moving, produce com affine

#rigid transform (dipy); from static and moving and com affine, produce rigid affine

#affine transform (dipy); from static and moving and rigid affine, produce final affine


## code_our_version.py [Zuzanna]

#resample (our version); from static and moving, produce new affine

#center of mass transform (our version); from static and moving, produce com affine

#rigid transform (our version); from static and moving and com affine, produce rigid affine

#affine transform (our version); from static and moving and rigid affine, produce final affine

#to help with optimization: gaussian pyramid resampling to use for each transform

#### TESTS TO WRITE

## test_sanity.py **to check our code for errors** [Michael]

#make fake data with known transforms
    # - func vol0 to moved vol0 (identical image) : to test if data loading, should be easy
    # - func vol0 to (moved) vol1 (same kind of image) : to test if robust to slight variation, should be easy
    # - func vol0 to (moved) negative vol0 (different kind of image) **maybe better version of this exists? : to test mutual information usefulness

#check all func from code_from_dipy.py and code_our_version.py

#record run time, accuracy, etc. (useful for comparing dipy to our version later)

## test_compare.py **to compare our restults to existing package; interesting problems** [figure out later]

#using real data, expand on comparisons from test_sanity.py

#contribution of gaussian pyramid?

#add more here as we build our funcs
