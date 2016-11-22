""" py.test test for pipeline.py

Run with:

    py.test test_pipeline.py
"""

import numpy as np
from fmri_utils import anatomical_preprocess, functional_preprocess, segmentation, anatomical_reg, functional_reg

def fake_T1_raw(I,J,K):
    T1_raw = np.ones((I,J,K))
    return T1_raw

def fake_pop_priors(I,J,K):
    pop_priors = np.ones((I,J,K))
    return pop_priors

def fake_EPI_raw(I,J,K,T):
    EPI_raw = np.ones((I,J,K,T))
    return EPI_raw

def fake_MNI(I,J,K):
    MNI = np.ones((I,J,K))
    return MNI

def test_anatomical_preprocess():
    #generate fake data for test
    [I,J,K,T] = [4,5,6,7]
    T1_raw = fake_T1_raw(I,J,K)
    pop_priors = fake_pop_priors(I,J,K)

    #get func outputs
    T1_skull, T1_whole, T1_mask = anatomical_preprocess(T1_raw, pop_priors)

    #check shape of outputs
    assert(T1_skull.shape==(I,J,K))
    assert(T1_whole.shape==(I,J,K))
    assert(T1_mask.shape==(I,J,K))

def test_functional_preprocess():
    #generate fake data for test
    [I,J,K,T] = [4,5,6,7]
    EPI_raw = fake_EPI_raw(I,J,K,T)

    #get func outputs
    EPI_cor, motion_params, EPI_mean, EPI_mask = functional_preprocess(EPI_raw)

    #check shape of outputs
    assert(EPI_cor.shape==(I,J,K,T))
    assert(motion_params.shape[0]==T)
    assert(EPI_mean.shape==(I,J,K))
    assert(EPI_mask.shape==(I,J,K))

def test_segmentation():
    #generate fake data for test
    [I,J,K,T] = [4,5,6,7]
    T1_raw = fake_T1_raw(I,J,K)
    pop_priors = fake_pop_priors(I,J,K)
    T1_skull, T1_whole, T1_mask = anatomical_preprocess(T1_raw, pop_priors)

    #get func outputs
    T1_WMprob, T1_GMprob, T1_CSFprob = segmentation(T1_skull)

    #check shape of outputs
    assert(T1_WMprob.shape==(I,J,K))
    assert(T1_GMprob.shape==(I,J,K))
    assert(T1_CSFprob.shape==(I,J,K))

def test_anatomical_reg():
    #generate fake data for test
    [I,J,K,T] = [4,5,6,7]
    T1_raw = fake_T1_raw(I,J,K)
    pop_priors = fake_pop_priors(I,J,K)
    MNI = fake_MNI(I,J,K)
    T1_skull, T1_whole, T1_mask = anatomical_preprocess(T1_raw, pop_priors)

    #get func outputs
    T1_in_MNI, T1_x2_MNI = anatomical_reg(T1_whole, MNI)

    #check shape of outputs
    assert(T1_in_MNI.shape==(I,J,K))
    assert(T1_x2_MNI.shape==(4,4))

def test_functional_reg():
    #generate fake data for test
    [I,J,K,T] = [4,5,6,7]
    T1_raw = fake_T1_raw(I,J,K)
    EPI_raw = fake_EPI_raw(I,J,K,T)
    pop_priors = fake_pop_priors(I,J,K)
    MNI = fake_MNI(I,J,K)
    T1_skull, T1_whole, T1_mask = anatomical_preprocess(T1_raw, pop_priors)
    EPI_cor, motion_params, EPI_mean, EPI_mask = functional_preprocess(EPI_raw)
    T1_in_MNI, T1_x2_MNI = anatomical_reg(T1_whole, MNI)

    #get func outputs
    EPI_corrected_in_MNI, EPI_mean_in_MNI, EPI_x2_MNI = functional_reg(EPI_cor, EPI_mean, T1_x2_MNI)

    #check shape of outputs
    assert(EPI_corrected_in_MNI.shape==(I,J,K,T))
    assert(EPI_mean_in_MNI.shape==(I,J,K))
    assert(EPI_x2_MNI.shape==(4,4))
