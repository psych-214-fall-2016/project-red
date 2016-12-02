""" py.test test for data_hashes.py

Run with:

    py.test test_code_our_version.py
"""

from fmri_utils.data_hashes import validate_data

def test_validate_data():

    data_directory = "../../../data"
    hash_file = "data_hashes.txt"

    validate_data(data_directory, hash_file)
