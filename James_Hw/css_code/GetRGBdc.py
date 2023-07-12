#12/06/2012
#Jun Jiang
#Summary
#   The fucntion is to get the digital counts from the captured image (a
#   CCDC target)
#
#[IN]
#   folder: the folder within which captured data is saved
#   bayerP: the bayer pattern of the raw data
#
#[OUT]
#   radiance: the radiance (raw data) of each patch in the CCDC target
#
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy
    
def GetRGBdc(folder = None,bayerP = None): 
    ## load the raw data
    
    if (len(varargin) == 0):
        print('folder has to be specified')
        return radiance
    else:
        if (len(varargin) == 1):
            print('bayer pattern has to be specified. use dcraw -i -v ')
    
    scipy.io.loadmat(np.array([folder,'rawData.mat']))
    ## subtract the dark image
    img = double(img)
    # get the dark image taken with lens cap on
    imgDark = imread(np.array([folder,'./canon60d_black.pgm']))
    img2 = img - double(imgDark)
    img2[img2 < 0] = 0
    ## No demosaic is used
    if (str(bayerP) == str('RGGB')):
        imgR = img2(np.arange(1,end()+2,2),np.arange(1,end()+2,2))
        imgG = img2(np.arange(1,end()+2,2),np.arange(2,end()+2,2))
        imgB = img2(np.arange(2,end()+2,2),np.arange(2,end()+2,2))
    else:
        if (str(bayerP) == str('GBRG')):
            imgR = img2(np.arange(2,end()+2,2),np.arange(1,end()+2,2))
            imgG = img2(np.arange(1,end()+2,2),np.arange(1,end()+2,2))
            imgB = img2(np.arange(1,end()+2,2),np.arange(2,end()+2,2))
        else:
            if (str(bayerP) == str('BGGR')):
                imgR = img2(np.arange(2,end()+2,2),np.arange(2,end()+2,2))
                imgG = img2(np.arange(1,end()+2,2),np.arange(2,end()+2,2))
                imgB = img2(np.arange(1,end()+2,2),np.arange(1,end()+2,2))
            else:
                if (str(bayerP) == str('GRBG')):
                    imgR = img2(np.arange(1,end()+2,2),np.arange(2,end()+2,2))
                    imgG = img2(np.arange(1,end()+2,2),np.arange(1,end()+2,2))
                    imgB = img2(np.arange(2,end()+2,2),np.arange(1,end()+2,2))
    
    ## extract the four corners of the CCDC
#Click on the four corners of CCDC in the order of (top left, top right,
#bottom right, and bottom left)
#You only need to do this once. To re-select, delete the file xyCorner.mat
#in the folder first
    
    xyCornerFile = 'xyCorner.mat'
    if (len(dir(np.array([folder,xyCornerFile])))==0):
        imagesc(imgG)
        grid('on')
        xyCorner = ginput(4)
        save(np.array([folder,xyCornerFile]),'xyCorner')
    else:
        scipy.io.loadmat(np.array([folder,xyCornerFile]))
    
    xyCorner = np.round(xyCorner)
    xyCorner = flipdim(xyCorner,2)
    rowRange = np.arange(np.amin(xyCorner(np.arange(1,2+1),1)),np.amin(xyCorner(np.arange(3,4+1),1))+1)
    colRange = np.arange(np.amin(xyCorner(np.array([1,4]),2)),np.amax(xyCorner(np.array([2,3]),2))+1)
    imgR = imgR(rowRange,colRange)
    imgG = imgG(rowRange,colRange)
    imgB = imgB(rowRange,colRange)
    figure
    imagesc(imgR)
    ## get the coordinates of each patch
    figure
    imagesc(imgR)
    grid('on')
    nRow = 12
    nCol = 20
    patchSz = np.array([imgR.shape[1-1] / nRow,imgR.shape[2-1] / nCol])
    col = np.arange(patchSz(2) / 2,imgG.shape[2-1]+patchSz(2),patchSz(2))
    row = np.arange(patchSz(1) / 2,imgG.shape[1-1]+patchSz(1),patchSz(1))
    col = np.round(col)
    row = np.round(row)
    hold('on')
    plt.plot(col,np.matlib.repmat(row,len(col),1),'ko')
    ##
    patchSamplingSz = 4
    radiance = np.zeros((3,nRow * nCol))
    radiance[1,:] = GetPatchRadiance(imgR,row,col,patchSamplingSz)
    radiance[2,:] = GetPatchRadiance(imgG,row,col,patchSamplingSz)
    radiance[3,:] = GetPatchRadiance(imgB,row,col,patchSamplingSz)
    return radiance
    
    return radiance