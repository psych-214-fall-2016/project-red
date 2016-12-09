#!/usr/bin/env python3
''' Installation script for fmri_utils package '''
from os.path import join as pjoin
from glob import glob

import setuptools

from distutils.core import setup

setup(name='fmri_utils',
      description='Functions for psych-214 class project',
      packages=['fmri_utils', 'fmri_utils.func_preproc', 'fmri_utils.registration',
      'fmri_utils.segmentation'],
      license='BSD license',
      package_data = {'fmri_utils': [pjoin('tests', '*')]},
      # Add all the scripts in the scripts directory
      scripts = glob(pjoin('scripts', '*')),
      )
