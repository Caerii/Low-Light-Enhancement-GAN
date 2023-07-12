import sys
# /Users/jamesau/Documents/GitHub/Math_Tool
sys.path.insert(0, '/Users/jamesau/Documents/GitHub/Image Processing use')
import trans
import cv2
import numpy as np
import matplotlib.pyplot as plt

maxnum = 255

def cal_height(inten,ref):
    arr = []
    preval = 0
    for i in range(len(inten)):
        num = 0
        while True:
            ade = ((inten[i]+preval)/2)
            if num+1 >= maxnum:
                arr.append(maxnum)
                preval = inten[i]
                break
            if ade <= ref[num]:
                arr.append(num)
                preval = inten[i]
                break
            num += 1
    return arr



def Histogram_equalization_modified(img_entry,img_reference,maxnum):
    img = np.array(img_entry)
    # img = np.array([[12,24,34,35,64],[12,24,34,35,64],[12,24,34,35,64],[12,24,34,35,64],[12,24,34,35,64]])
    histo, bin_edges = np.histogram(img.flatten(), bins=list(range(maxnum+1)))
    # equ = cv2.equalizeHist(img)
    cdf = histo.cumsum()
    cdf_normalized = cdf/cdf.max()#* float(histo.max()) / cdf.max()
    # sval = [i*(maxnum-1) for i in cdf_normalized]
    # sval_rnd = [round(i) for i in sval]

    img_ref = np.array(img_reference)
    histo2, bin_edges2 = np.histogram(img_ref.flatten(), bins=list(range(maxnum+1)))
    cdf2 = histo2.cumsum()
    cdf_normalized2 = cdf2/cdf2.max()
    # print(cdf_normalized,cdf_normalized2)
    # print(min(map(min, img_entry)),max(map(max, img_entry)))
    # print(len(cdf_normalized),len(cdf_normalized2))
    cdf_normalized = np.insert(cdf_normalized,0,0)
    cdf_normalized2 = np.insert(cdf_normalized2,0,0)
    # print(len(cdf_normalized),len(cdf_normalized2))
    sval = cal_height(cdf_normalized,cdf_normalized2)
    # print(sval)
    newimg = []
    for rowItem in img:
        row = []
        for p in rowItem:
            newnum = round(sval[p])
            row.append(newnum)
        newimg.append(row)
    # return sval_rnd,newimg
    return newimg



""" WorkFlow Control"""
itemsLink = 'Pics/Nyhavn.jpg'#"lena15.jpeg"
# itemsLink = input("EnterLink:")
img = cv2.imread(itemsLink,0)
imref = cv2.imread('Pics/NYCSunset.jpg',0)
imgout=Histogram_equalization_modified(img,imref,maxnum)
cv2.imwrite('Ordinary_result.png',np.array(imgout))
# plt.imshow(imgout)
# plt.show()
# result =cv2.equalizeHist(img)