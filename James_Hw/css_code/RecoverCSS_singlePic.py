#06/12/2012
#
#Summary
#   The function is to estimate the camera spectral sensitivity under
#   the daylight whose spectrum is unknown
#   if debug is set to 1, the measured daylight and camera spectral
#   sensitivity of Canon 60D is used as an example. To use your own data,
#   turn debug to 0
#
#[OUT]
#   cmf: the ground truth (the measured camera spectral sensitivity)
#   cmfHat: the estimated camera spectral sensitivity
#
import numpy as np
import matplotlib.pyplot as plt
import scipy

def RecoverCSS_singlePic(): 
    ## Load captured CCDC
    
    # load info of captured images
    folder = './raw/'
    filename = 'img_0153'
    # Convert from CR2 to pgm
    system(np.array(['./dcraw -4 -D -j -v -t 0 ',np.array([folder,filename,'.CR2'])]))
    # save as mat file
    img = imread(np.array([folder,filename,'.pgm']))
    save(np.array([folder,'rawData.mat']),'img')
    bayerP = 'RGGB'
    ## Load the CCDC relectance (the duplicate and glossy patches are removed)
    wWanted = np.arange(400,720+10,10)
    w2 = np.arange(400,720+10,10)
    reflectance = scipy.io.loadmat('CCDC_meas.mat')
    reflectance = reflectance.CCDC_meas
    reflectance = reflectance.spectra
    glossyP = np.array([79,99,119,139,159,179,199,219])
    darkP = np.array([21,40,81,100,141,160,201,220])
    darkP = np.array([darkP,150,151,152])
    unwantedP = np.array([glossyP,darkP])
    refl = reflectance(np.arange(3,end() - 1+1),:)
    w = w2
    #remove the duplicate and glossy patches in CCDC
    range_ = np.arange(21,220+1)
    refl2 = np.zeros((len(w),len(range_) - len(unwantedP)))
    idx = 1
    for i in np.arange(range_(1),range_(end())+1).reshape(-1):
        if (len(find(unwantedP == i))==0):
            refl2[:,idx] = refl(:,i)
            idx = idx + 1
    
    clear('refl')
    refl = refl2
    clear('refl2')
    ## Load captured radiance by the camera
# the raw data image are captured and saved as mat file
    
    radiance1 = GetRGBdc(folder,bayerP)
    radiance1 = radiance1 / (2 ** 16)
    # remove the radiance of those glossy or duplicate patches in CCDC
    range_ = np.arange(21,220+1)
    radiance1Copy = np.zeros((3,len(range_) - len(unwantedP)))
    idx = 1
    for i in np.arange(range_(1),range_(end())+1).reshape(-1):
        if (len(find(unwantedP == i))==0):
            radiance1Copy[:,idx] = radiance1(:,i)
            idx = idx + 1
    
    radiance1 = radiance1Copy
    radiance = radiance1
    clear('radiance1Copy')
    radiance = np.transpose(radiance)
    ## Load measured daylight for evaluation purpose only
    
    scipy.io.loadmat(np.array([folder,'daylight.mat']))
    w = np.arange(380,780+5,5)
    ill_groundTruth = interp1(w,ill,wWanted)
    ill_groundTruth = np.multiply(ill_groundTruth,100.0) / ill_groundTruth(find(wWanted == 560))
    clear('spd')
    w = wWanted
    ## Load the measured cmf (ground truth) of the camera
    camName = 'Canon60D'
    rgbCMF,camNameAll = getCameraSpectralSensitivity()
    for i in np.arange(1,len(camNameAll)+1).reshape(-1):
        if (str(camNameAll[i]) == str(camName)):
            cmf = np.array([rgbCMF[0](:,i),rgbCMF[2](:,i),rgbCMF[3](:,i)])
    
    cmf = cmf / np.amax(cmf)
    figure
    plt.plot(w,cmf(:,1),'r')
    hold('on')
    plt.plot(w,cmf(:,2),'g')
    hold('on')
    plt.plot(w,cmf(:,3),'b')
    plt.title('Measured camera response function')
    plt.legend('R','G','B')
    ## Get the eigenvectors of the camera spectral sensitivity
    numEV = 2
    eRed,eGreen,eBlue = PCACameraSensitivity(numEV)
    ## Recover camera spectral sensitivity from a single image under unknown daylight
    CCTrange = np.arange(4000,27000+100,100)
    diff_b = np.zeros((1,len(CCTrange)))
    for i in np.arange(1,len(CCTrange)+1).reshape(-1):
        ill = getDaylightScalars(CCTrange(i))
        deltaLamda = 10
        cmfHat[:,1] = RecoverCMFev(ill,refl,w,radiance(:,1),eRed)
        cmfHat[:,2] = RecoverCMFev(ill,refl,w,radiance(:,2),eGreen)
        cmfHat[:,3] = RecoverCMFev(ill,refl,w,radiance(:,3),eBlue)
        I_hat = np.transpose(refl) * diag(ill) * cmfHat * deltaLamda
        diff_b[i] = norm(radiance - I_hat)
    
    figure
    plt.plot(CCTrange,diff_b,'-o')
    plt.xlim(np.array([CCTrange(1),CCTrange(end())]))
    plt.xlabel('CCT')
    plt.ylabel('norm of radiance difference')
    minDiff,minDiffIdx = np.amin(diff_b)
    ill = getDaylightScalars(CCTrange(minDiffIdx))
    ill = ill / ill(find(w == 560))
    w = wWanted
    figure
    ill_groundTruth = ill_groundTruth / ill_groundTruth(find(w == 560))
    plt.plot(w,ill_groundTruth)
    hold('on')
    plt.plot(w,ill,'r-.')
    plt.legend('measured daylight','our result')
    cmfHat[:,1] = RecoverCMFev(ill,refl,w,radiance(:,1),eRed)
    cmfHat[:,2] = RecoverCMFev(ill,refl,w,radiance(:,2),eGreen)
    cmfHat[:,3] = RecoverCMFev(ill,refl,w,radiance(:,3),eBlue)
    cmfHat = cmfHat / np.amax(cmfHat)
    cmfHat[cmfHat < 0] = 0
    w = np.arange(400,720+10,10)
    figure
    plt.plot(w,cmf(:,1),'r')
    hold('on')
    plt.plot(w,cmf(:,2),'g')
    hold('on')
    plt.plot(w,cmf(:,3),'b')
    hold('on')
    plt.plot(w,cmfHat(:,1),'r-.')
    hold('on')
    plt.plot(w,cmfHat(:,2),'g-.')
    hold('on')
    plt.plot(w,cmfHat(:,3),'b-.')
    hold('on')
    plt.legend('r_m','g_m','b_m','r_e','g_e','b_e')
    save(np.array(['cmf',camName,'.mat']),'cmf','cmfHat')
    return cmfHat
    
    return cmfHat