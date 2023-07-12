#06/12/2012
#Chris
#Summary
#   The function is to recover the camera spectral sensitivity given the
#   spectral reflectance of the samples, the eigenvectors of the camera
#   sensitivity, and the illuminant spectrum
#
#[IN]
#   ill: the light source spectrum
#   reflSet: the spectral reflectance of samples
#   w: wavelength range
#   XYZSet: the radiance captured by the camera
#   e: eigenvector of the camera spectral sensitivity
#
#[OUT]
#   X: the recovered camera spectral sensitivity
#   A, b: Ax=b
#
import numpy as np
    
def RecoverCMFev(ill = None,reflSet = None,w = None,XYZSet = None,e = None): 
    numRefl = reflSet.shape[2-1]
    A = np.zeros((numRefl,e.shape[2-1]))
    b = np.zeros((A.shape[1-1],1))
    deltaLambda = 10
    weight = 1
    for i in np.arange(1,numRefl+1).reshape(-1):
        #weight=XYZSet(i);
        A[i,:] = np.multiply(np.multiply(np.transpose(reflSet(:,i)) * diag(ill) * e,deltaLambda),weight)
        b[i] = XYZSet(i) * weight
    
    X = np.linalg.solve(A,b)
    #X = lsqnonneg(A,b);
    
    X = e * X
    return X,A,b
    
    return X,A,b