"""
general functions, useful for other registration methods and tests

"""

import nibabel as nib


def get_data_affine(fname):
    """ get data and affine from filename

    Parameters
    ----------
    fname : str
        *.nii or *.nii.gz file name

    Returns
    -------
    data : array shape (I, J, K) or (I, J, K, T)
        array with 3D or 4D data from image file
    affine : array shape (4, 4)
        affine for image file
    """
    img = nib.load(fname)
    data = img.get_data()
    affine = img.get_affine()

    return (data, affine)
