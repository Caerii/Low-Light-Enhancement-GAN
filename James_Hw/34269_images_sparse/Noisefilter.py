import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy
import skimage 
# import skimage

# import trans
def process(image):
    rows,cols = image.shape
    kernel1 = np.ones((10, 10), np.uint8)
    image1=cv2.GaussianBlur(image,(5,5),0)
    noise = np.zeros(image.shape, image.dtype)
    m = (15, 15, 15)
    s = (30, 30, 30)
    cv2.randn(noise, m, s)
    image = cv2.add(image, noise)

""" WorkFlow Control"""
# itemsLink = "/Users/jamesau/Documents/GitHub/Image Processing use/how-to-remove-censored-part-3.jpg"#"lena15.jpeg"
itemsLink = input("EnterLink:")
img = cv2.imread(itemsLink,0)
img2,imgD,imgE,imgF = process(img)