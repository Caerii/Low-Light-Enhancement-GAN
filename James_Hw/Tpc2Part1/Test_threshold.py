import sys
# /Users/jamesau/Documents/GitHub/Math_Tool
sys.path.insert(0, '/Users/jamesau/Documents/GitHub/Image Processing use')
import trans
import cv2
import numpy as np
import matplotlib.pyplot as plt
def process(image):
    rows,cols = image.shape
    ret = trans.threshold(image,rows,cols,0.5*255,0,1*255)
    return ret
""" WorkFlow Control"""
itemsLink = "Pics/goldhill.png"#"lena15.jpeg"
# itemsLink = input("EnterLink:")
img = cv2.imread(itemsLink,0)
imgout = process(img)
plt.imshow(imgout)
# plt.imshow('Image',np.array())
plt.show()
# cv2.
