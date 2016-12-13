""" py.test test for data_hashes.py

Run with:

    py.test test_code_our_version.py
"""
import os
from os.path import join as pjoin, dirname
from fmri_utils.check_data_org import safe_make_dir, make_nii_hashes_nested, validate_data, main
import random, string

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
    hash_file = "temp.txt"

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

    validate_data(data_directory, hash_file)

    # rewrite hash_file with only sample_file line; filename intentionally wrong
    fake_file = 'temp_img.nii'
    fake_line = " ".join([split_lines[sample_idx[0]][0], fake_file])

    hobj = open(pjoin(data_directory,hash_file), 'wt')
    hobj.write(fake_line)
    hobj.close()

    try:
        validate_data(data_directory, hash_file)
    except FileNotFoundError:
        print('bad file name caught successfully!')

    # rewrite hash_file with only sample_file line; hash intentionally broken
    fake_hash = "".join([random.choice(string.digits + 'abcdef') for n in range(40)])
    fake_line = " ".join([fake_hash, split_lines[sample_idx[0]][1]])

    kobj = open(pjoin(data_directory,hash_file), 'wt')
    kobj.write(fake_line)
    kobj.close()

    try:
        validate_data(data_directory, hash_file)
    except ValueError:
        print('bad hash caught successfully!')

    # remove sample hash_file
    os.remove(pjoin(data_directory,hash_file))
    assert(not os.path.exists(pjoin(data_directory,hash_file))) # dir doesn't exist

def test_main():
    try:
        main()
    except FileNotFoundError:
        print('testing main; should fail because data missing img files')
