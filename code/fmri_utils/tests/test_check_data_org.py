""" py.test test for data_hashes.py

Run with:

    py.test test_code_our_version.py
"""
import os
from os.path import join as pjoin, dirname
from fmri_utils.check_data_org import safe_make_dir, make_nii_hashes_nested, validate_data

def test_safe_make_dir():

    # check that temporary dir doesn't exit
    tempdir = 'tempdir_for_test'
    assert(not os.path.isdir(tempdir))

    # try to make temporary dir
    try1 = safe_make_dir(tempdir)
    assert(os.path.isdir(tempdir)) # dir exists
    assert(try1) # dir created

    # try to make temporary dir again
    try2 = safe_make_dir(tempdir)
    assert(os.path.isdir(tempdir)) # dir stil exists
    assert(not try2) # dir NOT created

    # remove temporary dir
    os.rmdir(tempdir)
    assert(not os.path.isdir(tempdir)) # dir doesn't exist

def test_hash_sample():

    # test extraction and generation of hash for MNI template file
    data_directory = pjoin(dirname(__file__))
    hash_file = "test_data_hashes.txt"

    # make sure file doesn't already exist
    assert(not os.path.exists(pjoin(data_directory, hash_file))) # dir doesn't exist

    # make hash file with existing file
    make_nii_hashes_nested(data_directory, hash_file)

    # check that sample file is in hash_file
    sample_file = 'ds107_sub012_t1r2_small.nii'

    fobj = open(pjoin(data_directory,hash_file), 'rt')
    lines = fobj.readlines()
    fobj.close()
    split_lines= [line.split() for line in lines]
    sample_idx = [i for i in range(len(split_lines)) if split_lines[i][1].find(sample_file)>-1]

    assert(len(sample_idx)==1) # one row with correct file

    # rewrite hash_file with only sample_file line

    gobj = open(pjoin(data_directory,hash_file), 'wt')
    gobj.write(" ".join(split_lines[sample_idx[0]]))
    gobj.close()

    # Call function to validate data in data directory
    validate_data(data_directory, hash_file)

    # remove sample hash_file
    os.remove(pjoin(data_directory,hash_file))
    assert(not os.path.exists(pjoin(data_directory,hash_file))) # dir doesn't exist
