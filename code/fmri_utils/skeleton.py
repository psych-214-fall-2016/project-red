"""
skeleton.py:
outline for preprocessing project
--> use as temp organizaiton tool, don't have to keep
includes functions for each module (corect input and output vars types,
    but no calculations yet) and tests (for output var type, otherwise garbage)
"""
import numpy as np


def anatomical_preprocess(T1_raw, pop_priors):
    """ Do anatomical preprocessing on `T1_raw` using `pop_priors`

    Parameters
    ----------
    T1_raw : array shape (I, J, K)
        3D array containing T1 contrast image values
    pop_priors : array ahape (I, J, K)
        3D array containing population priors for T1 contrast

    Returns
    -------
    T1_skullstripped : array shape (I, J, K)
        Deobliqued, RPI reoriented, bias-corrected, skull-stripped T1
    T1_wholehead : array shape (I, J, K)
        Deobliqued, RPI reoriented, bias-corrected, whole-head T1
    T1_brainmask : array shape (I, J, K)
        Deobliqued, RPI reoriented, bias-corrected, T1 brain vox mask
    """

    T1_skullstripped = np.ones(T1_raw.shape)
    T1_wholehead = np.ones(T1_raw.shape)
    T1_brainmask = np.ones(T1_raw.shape)

    return (T1_skullstripped, T1_wholehead, T1_brainmask)


def functional_preprocess(EPI_raw):
    """ Do functional preprocessing on `EPI_raw`

    Parameters
    ----------
    EPI_raw : array shape (I, J, K, T)
        4D array containing multiple (T) 3D EPI volumes


    Returns
    -------
    EPI_corrected : array shape (I, J, K, T)
        Deobliqued, RPI reoriented, slice-time and motion corrected EPI
    motion_params : array shape (T, N)
        rigid-body motion parameters for T volumes in N dimensions
    EPI_mean : array shape (I, J, K)
        mean EPI image of EPI_corrected
    EPI_mask : array shape (I, J, K)
        brain vox mask of EPI_mean
    """

    EPI_corrected = np.ones(EPI_raw.shape)
    N = 6
    motion_params = np.ones((EPI_raw.shape[-1], N))
    EPI_mean = np.ones(EPI_raw.shape[:3])
    EPI_mask = np.ones(EPI_raw.shape[:3])

    return (EPI_corrected, motion_params, EPI_mean, EPI_mask)


def segmentation(T1_skullstripped):
    """ Do tissue segmentation on `T1_skullstripped`

    Parameters
    ----------
    T1_skullstripped : array shape (I, J, K)
        Deobliqued, RPI reoriented, bias-corrected, skull-stripped T1

    Returns
    -------
    T1_WMprob : array shape (I, J, K)
        probability map of T1 white matter
    T1_GMprob : array shape (I, J, K)
        probability map of T1 gray matter
    T1_CSFprob : array shape (I, J, K)
        probability map of T1 cerebral spinal fluid
    """

    T1_WMprob = np.ones(T1_skullstripped.shape)
    T1_GMprob = np.ones(T1_skullstripped.shape)
    T1_CSFprob = np.ones(T1_skullstripped.shape)

    return (T1_WMprob, T1_GMprob, T1_CSFprob)

def anatomical_reg(T1_wholehead, MNI):
    """ Do anatomical registration of `T1_wholehead` to `MNI` template

    Parameters
    ----------
    T1_wholehead : array shape (I, J, K)
        Deobliqued, RPI reoriented, bias-corrected, whole-head T1
    MNI : array shape(I, J, K)
        MNI T1 template

    Returns
    -------
    T1_in_MNI : array shape (I, J, K)
        T1_wholehead transformed to MNI space
    T1_x2_MNI : array shape (4, 4)
        affine for T1 transform to MNI space
    """

    T1_in_MNI = np.ones(T1_wholehead.shape)
    T1_x2_MNI = np.eye(4)

    return (T1_in_MNI, T1_x2_MNI)


def functional_reg(EPI_corrected, EPI_mean, T1_x2_MNI):
    """ Do functional registration of `EPI_mean` (and `EPI_corrected`)
        to `MNI` template

    Parameters
    ----------
    EPI_corrected : array shape (I, J, K, T)
        Deobliqued, RPI reoriented, slice-time and motion corrected EPI
    EPI_mean : array shape (I, J, K)
        mean EPI image of EPI_corrected
    T1_x2_MNI : array shape (4, 4)
        affine for T1 transform to MNI space

    Returns
    -------
    EPI_corrected_in_MNI : array shape (I, J, K, T)
        EPI_corrected transformed to MNI space
    EPI_mean_in_MNI : array shape (I, J, K)
        EPI_mean transformed to MNI space
    EPI_x2_MNI : array shape (4, 4)
        affine for EPI_mean transform to MNI space
    """

    EPI_corrected_in_MNI = np.ones(EPI_corrected.shape)
    EPI_mean_in_MNI = np.ones(EPI_mean.shape)
    EPI_x2_MNI = np.eye(4)

    return (EPI_corrected_in_MNI, EPI_mean_in_MNI, EPI_x2_MNI)
