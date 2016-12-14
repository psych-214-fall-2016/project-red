"""
verify that data organization is correct and all necessary files are present

following instructions from project-red/README.md


"""
import os
from os.path import dirname, join as pjoin, isdir
import hashlib

def safe_make_dir(dir_path):
    """ make dir if doesn't exist already

    Parameters
    ----------
    dir_path : str
        Name of dir to create

    Returns
    -------
    created : bool
        if created new dir

    """

    if isdir(dir_path):
        return False
    else:
        os.mkdir(dir_path)
        return True


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


def make_nii_hashes_nested(data_directory, hash_file):
    """ Generate hashes for all *.nii* in `data_directory`
        find all *.nii* nested in dirs
        save each <hash> <file_path> in `hash_file`

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

    dir_elements = os.listdir(data_directory)

    # collect all *.nii* files in `data_directory`
    nii_files = [f for f in dir_elements if f.find('.nii')>-1]

    # collect all sub directories
    sub_dirs = [f for f in dir_elements if isdir(pjoin(data_directory,f))]

    # do same for each sub directory until end of tree
    while len(sub_dirs)>0:
        next_dirs = []

        for s in sub_dirs:
            dir_elements = os.listdir(pjoin(data_directory,s))
            nii_files.extend([pjoin(s,f) for f in dir_elements if f.find('.nii')>-1])
            next_dirs.extend([pjoin(s,f) for f in dir_elements if isdir(pjoin(data_directory,s,f))])
        sub_dirs = next_dirs

    # generate hashes in save each file info in `hash_file`
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

    data_directory = pjoin(dirname(__file__),'../../data')
    hash_file = "data_hashes.txt"

    # # Generate hash_file for all *.nii* files in data directory
    # make_nii_hashes_nested(data_directory, hash_file)

    # Call function to validate data in data directory
    validate_data(data_directory, hash_file)



if __name__ == '__main__':
    # Python is running this file as a script, not importing it.

    main()
