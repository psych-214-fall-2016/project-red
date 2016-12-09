"""get hashes for files in data folder, validate

Run as:
    python3 validate_data.py ../../data
"""

import os
from os.path import join as pjoin
import sys
import hashlib

def file_hash(filename):
    """ Get byte contents of file `filename`, return SHA1 hash

    Parameters
    ----------
    filename : str
        Name of file to read

    Returns
    -------
    hash : str
        SHA1 hexadecimal hash string for contents of `filename`.
    """
    # Open the file, read contents as bytes.
    fobj = open(filename, 'rb')
    contents = fobj.read()
    fobj.close()

    # Calculate, return SHA1 has on the bytes from the file.
    return hashlib.sha1(contents).hexdigest()

def make_nii_hashes(data_directory, hash_file):
    """ Generate hashes for all nii files in `data_directory`, save as output_file

    Parameters
    ----------
    data_directory : str
        Directory containing data; will save output_file here

    hash_file : str
        File with hashes for expected data files; text file containing hashes of form <hash> <filename>

    Returns
    -------
    None

    """

    all_files = os.listdir(data_directory)
    nii_files = [f for f in all_files if f.find('.nii')>0]

    g = open(pjoin(data_directory,hash_file),'w')

    for f in nii_files:
        hash_str = file_hash(pjoin(data_directory,f))
        g.write(" ".join([hash_str,f,'\n']))

    g.close()

def validate_data(data_directory, hash_file):
    """ Read ``data_hashes.txt`` file in `data_directory`, check hashes

    Parameters
    ----------
    data_directory : str
        Directory containing data and ``data_hashes.txt`` file.

    hash_file : str
        File with hashes for expected data files; text file containing hashes of form <hash> <filename>

    Returns
    -------
    None

    Raises
    ------
    ValueError:
        If hash value for any file is different from hash value recorded in
        ``data_hashes.txt`` file.
    """
    # Read lines from ``data_hashes.txt`` file.
    fobj = open(pjoin(data_directory,hash_file), 'rt')
    lines = fobj.readlines()
    fobj.close()

    # Split into SHA1 hash and filename
    split_lines = [line.split() for line in lines]

    # Calculate actual hash for given filename.
    for line in split_lines:

        fhash = file_hash(pjoin(data_directory,line[1]))

        # If hash for filename is not the same as the one in the file, raise
        # ValueError
        if fhash != line[0]:
            raise ValueError('Hash mismatch in file: ' + pjoin(data_directory,line[1]))

    print('Files validated.')
    return 1


def main():
    # Get the data directory from the command line arguments
    if len(sys.argv) < 2:
        raise RuntimeError("Please give data directory on "
                           "command line")
    data_directory = sys.argv[1]

    hash_file = "data_hashes.txt"

    # Generate hash_file for all *.nii* files in data directory
    #make_nii_hashes(data_directory, hash_file)

    # Call function to validate data in data directory
    validate_data(data_directory, hash_file)



if __name__ == '__main__':
    # Python is running this file as a script, not importing it.
    main()
