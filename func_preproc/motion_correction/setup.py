#!/usr/bin/env python3
''' Installation script for motion_correction package '''
from os.path import join as pjoin
from glob import glob

import setuptools

from distutils.core import setup

setup(name='motion_correction',
      version='0.1',
      description='Tools for running the motion correction',
      packages=['glmtools'],
      license='BSD license',
      author='Your name here',
      author_email='yourname@berkeley.edu',
      maintainer='Your name here',
      maintainer_email='yourname@berkeley.edu',
      url='',
      package_data = {'motion_correction': [pjoin('tests', '*')]},
      # Add all the scripts in the scripts directory
      scripts = glob(pjoin('scripts', '*')),
      requires=['numpy (>=1.5.1)', 'scipy (>=0.16.0)'],
      )
