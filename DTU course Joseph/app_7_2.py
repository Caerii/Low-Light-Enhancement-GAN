import scipy
import cv2
import PIL
import random
import numpy as np 
img = cv2.imread('C:/Users/Josep/OneDrive/Desktop/DTU course/day 4/ExTopic6SegKpts/imSeg/Blob.tif')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude, direction = cv2.cartToPolar(sobel_x, sobel_y)
threshold_value = 50
edges = cv2.threshold(magnitude, threshold_value, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('Input Image', img)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
ksize = (5, 5)
sigma = random.uniform(49, 50)
noise = cv2.GaussianBlur(np.zeros_like(gray), ksize, sigma)
noisy_gray = cv2.add(gray, noise, dtype=cv2.CV_8UC1)

sobel_x = cv2.Sobel(noisy_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(noisy_gray, cv2.CV_64F, 0, 1, ksize=3)

magnitude, direction = cv2.cartToPolar(sobel_x, sobel_y)

threshold_value = 50
edges = cv2.threshold(magnitude, threshold_value, 255, cv2.THRESH_BINARY)[1]

cv2.imshow('Input Image', img)
cv2.imshow('Noisy Image', noisy_gray)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
