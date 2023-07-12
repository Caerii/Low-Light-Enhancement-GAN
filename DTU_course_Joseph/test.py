import cv2
import numpy as np
from PIL import Image

# Load an image
img = cv2.imread('C:/Users/Josep/OneDrive/Desktop/DTU course/day 4/ExTopic6SegKpts/imKpts/ND1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Compute the derivatives along the horizontal and vertical direction
Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Compute the components of the second order moment matrix M
Ix2 = Ix ** 2
Iy2 = Iy ** 2
IxIy = Ix * Iy

# Create an array structM with dimensions same as input image and third dimension 3
structM = np.zeros((gray.shape[0], gray.shape[1], 3))

# Assign the Ix2, Iy2, and IxIy to the third dimension of structM
structM[:, :, 0] = Ix2
structM[:, :, 1] = Iy2
structM[:, :, 2] = IxIy

# Apply a Gaussian filter on structM
ksize = (3, 3)
sigma = 1
structM = cv2.GaussianBlur(structM, ksize, sigma)

# Compute the cornerness measure at each pixel
k = 0.04
detM = (structM[:, :, 0] * structM[:, :, 1]) - (structM[:, :, 2] ** 2)
traceM = structM[:, :, 0] + structM[:, :, 1]
C = detM - k * (traceM ** 2)

# Threshold C
threshold_value = 0.0000001 * C.max()
C[C < threshold_value] = 0

# Apply non-maximum suppression to C
neighborhood_size = 3
C_max = cv2.dilate(C, np.ones((neighborhood_size, neighborhood_size)))
C[C < C_max] = 0

# Convert the corners to a PIL image
corners_pil = Image.fromarray(np.uint8(C))

# Display the result
corners_pil.show()