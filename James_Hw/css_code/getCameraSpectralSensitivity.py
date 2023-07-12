#07/25/2012
#Chris
#Summary
#   The function is to get all the camera spectral sensitivity from the
#   database of 28 cameras
#[OUT]
#   rgbCMF: the 1x3 cell containing the camera spectral sensitivity for
#   each channel
#   camName: camera names
#
import numpy as np
import scipy   
def getCameraSpectralSensitivity(): 
    folder = './camSpecSensitivity/'
    files = dir(folder)
    idx = 1
    for i in np.arange(1,len(files)+1).reshape(-1):
        if (len(files(i).name) > 5 and str(files(i).name(np.arange(1,3+1))) == str('cmf')):
            scipy.io.loadmat(np.array([folder,files(i).name]))
            redCMF[:,idx] = np.transpose(r)
            greenCMF[:,idx] = np.transpose(g)
            blueCMF[:,idx] = np.transpose(b)
            camName[idx] = files(i).name(np.arange(5,end() - 4+1))
            idx = idx + 1
    
    rgbCMF[0] = redCMF
    rgbCMF[2] = greenCMF
    rgbCMF[3] = blueCMF
    return rgbCMF,camName
    
    return rgbCMF,camName