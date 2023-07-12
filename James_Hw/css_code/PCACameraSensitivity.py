#06/06/2012
#
#Summary
#   The function is to do PCA on camera sensitivity
#[IN]
#   numEV: number of eigenvectors to retain
#
#[OUT]
#   eRed, eGreen, eBlue: the eigenvectors of each of the three channel
#
import numpy as np
    
def PCACameraSensitivity(numEV = None): 
    ## 07/30/2012
    
    rgbCMF = getCameraSpectralSensitivity()
    redCMF = rgbCMF[0]
    greenCMF = rgbCMF[2]
    blueCMF = rgbCMF[3]
    #normalize to each curve
    for i in np.arange(1,greenCMF.shape[2-1]+1).reshape(-1):
        redCMF[:,i] = redCMF(:,i) / np.amax(redCMF(:,i))
        greenCMF[:,i] = greenCMF(:,i) / np.amax(greenCMF(:,i))
        blueCMF[:,i] = blueCMF(:,i) / np.amax(blueCMF(:,i))
    
    ## do PCA on cmf
    
    if (len(varargin) > 0):
        retainEV = numEV
    else:
        retainEV = 1
    
    eRed = GetEigenvector(redCMF,retainEV)
    eGreen = GetEigenvector(greenCMF,retainEV)
    eBlue = GetEigenvector(blueCMF,retainEV)
    return eRed,eGreen,eBlue
    
    return eRed,eGreen,eBlue