""" py.test test for data_hashes.py

Run with:

    py.test test_code_our_version.py
"""
import os
from os.path import join as pjoin, dirname
from tempfile import TemporaryDirectory

import random, string

from fmri_utils.check_data_org import make_nii_hashes_nested, validate_data, main


def test_hash_sample():

    # move to temp dir so can save files
    with TemporaryDirectory() as tmpdirname:
        tempdir = tmpdirname

        # test extraction and generation of hash for MNI template file
        data_directory = pjoin(dirname(__file__))
        temp_file = pjoin(tempdir, 'temp_hash1.txt')

        # make hash file with existing file
        make_nii_hashes_nested(data_directory, temp_file)

        # check that sample file is in hash_file
        sample_file = 'ds107_sub012_t1r2_small.nii'

        fobj = open(pjoin(data_directory, temp_file), 'rt')
        lines = fobj.readlines()
        fobj.close()
        split_lines= [line.split() for line in lines]
        sample_idx = [i for i in range(len(split_lines)) if split_lines[i][1].find(sample_file)>-1]

        assert(len(sample_idx)==1) # one row with correct file

        # rewrite hash_file with only sample_file line)
        temp_file = pjoin(tempdir, 'temp_hash2.txt')
        gobj = open(pjoin(data_directory, temp_file), 'wt')
        gobj.write(" ".join(split_lines[sample_idx[0]]))
        gobj.close()

        validate_data(data_directory, temp_file)

        # rewrite hash_file with only sample_file line; filename intentionally wrong
        temp_file = pjoin(tempdir, 'temp_hash3.txt')
        fake_file = 'temp_img.nii'
        fake_line = " ".join([split_lines[sample_idx[0]][0], fake_file])

        hobj = open(pjoin(data_directory, temp_file), 'wt')
        hobj.write(fake_line)
        hobj.close()

        try:
            validate_data(data_directory, temp_file)
        except FileNotFoundError:
            print('bad file name caught successfully!')

        # rewrite hash_file with only sample_file line; hash intentionally broken
        temp_file = pjoin(tempdir, 'temp_hash4.txt')
        fake_hash = "".join([random.choice(string.digits + 'abcdef') for n in range(40)])
        fake_line = " ".join([fake_hash, split_lines[sample_idx[0]][1]])

        kobj = open(pjoin(data_directory, temp_file), 'wt')
        kobj.write(fake_line)
        kobj.close()

        try:
            validate_data(data_directory, temp_file)
        except ValueError:
            print('bad hash caught successfully!')

    # check that dir deleted
    assert(not os.path.isdir(tempdir))

def test_main():
    try:
        os.system('python3 ../check_data_org.py')
    except FileNotFoundError:
        print('testing main; should fail because data missing img files')
