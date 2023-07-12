import numpy as np
import cv2
filename = 'ND1.jpg'
# filename = 'chessboard.png'
img = cv2.imread(filename)
img_new = np.zeros(img.shape)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img_new[dst>0.01*dst.max()]=[0,0,255]
# img[dst>0.01*dst.max()]=[0,0,255]
# cv2.imshow('dst',img_new)
# # cv2.imshow('dst',img)
# if cv2.waitKey(0): #& 0xff == 27:
#     cv2.destroyAllWindows()
# 3x3 sobel filters for edge detection
# sobel_x = np.array([[ -1, 0, 1], 
#                     [ -2, 0, 2], 
#                     [ -1, 0, 1]])
# sobel_y = np.array([[ -1, -2, -1], 
#                     [  0,  0,  0], 
#                     [  1,  2,  1]])

# # Filter the blurred grayscale images using filter2D
# filtered_blurred_x = cv2.filter2D(img, cv2.CV_32F, sobel_x)  
# filtered_blurred_y = cv2.filter2D(img, cv2.CV_32F, sobel_y)

# mag = cv2.magnitude(filtered_blurred_x, filtered_blurred_y)
# orien = cv2.phase(filtered_blurred_x, filtered_blurred_y, angleInDegrees=True)
# cv2.imshow('dst',orien)
# # cv2.imshow('dst',img)
# if cv2.waitKey(0): #& 0xff == 27:
#     cv2.destroyAllWindows()
# sift = cv2.SIFT_create()
sift = cv2.xfeatures2d.SIFT_create()
# keypoints = sift.detect(gray,None)
# keypoints = sift.detect(img,None)
keypoints, descriptors = sift.detectAndCompute(img, None)
# draw the detected key points
sift_image = cv2.drawKeypoints(gray, keypoints, img)
# show the image
cv2.imshow('image', sift_image)
# save the image
# cv2.imwrite("table-sift.jpg", sift_image)
cv2.waitKey(0)
cv2.destroyAllWindows()