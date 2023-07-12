#12/06/2012
#Jun Jiang
#Summary
#   The function is to get the readiance of each patch based on the input paramter
#[IN]
#   img: the captured img
#   row: the number of rows
#   col: the number of cols of the patches
#   sampleSz: the window size within whcih pixel values are averaged
#[OUT]
#   radiance: the radiance of each patch
#
import numpy as np
    
def GetPatchRadiance(img = None,row = None,col = None,sampleSz = None): 
    if (np.mod(sampleSz,2)):
        sampleSz = sampleSz + 1
    
    #save in row major order
    radiance = np.zeros((1,len(row) * len(col)))
    for i in np.arange(1,len(row)+1).reshape(-1):
        for j in np.arange(1,len(col)+1).reshape(-1):
            pixels = img(np.arange(row(i) - sampleSz / 2,row(i) + sampleSz / 2+1),np.arange(col(j) - sampleSz / 2,col(j) + sampleSz / 2+1))
            radiance[[i - 1] * len[col] + j] = mean(pixels)
    
    return radiance
    
    # return radiance