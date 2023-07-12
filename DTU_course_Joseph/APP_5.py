import pandas as pd
import colour
import numpy as np
import PIL
import matplotlib.pyplot as plt
from skimage.util import random_noise
import pywt

# Load the image
img = PIL.Image.open(r'C:\Users\Josep\OneDrive\Desktop\DTU course\day3\34269_images_sparse\einstein.bmp')

# Convert to grayscale
imgarray = np.asarray(img)

# Apply Gaussian noise
mean = 0
variance = 100
sigma = np.sqrt(variance)
gaussian = np.random.normal(mean, sigma, imgarray.shape)
noisy_image = imgarray + gaussian
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Add additional Gaussian noise using scikit-image
noisy_image = random_noise(noisy_image, mode='gaussian', var=0.01)

# Display the original and noisy images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(noisy_image, cmap='gray')
ax[1].set_title('Gaussian Noise')
plt.show()


coeffs = pywt.dwt2(noisy_image, 'haar')
threshold = 4
coeffs_thresh = tuple(pywt.threshold(c, threshold, 'hard') for c in coeffs)


approx_coeffs = coeffs_thresh[0]
approx_coeffs_abs = np.abs(approx_coeffs)


reconstructed = pywt.idwt2((approx_coeffs_abs, coeffs[1]), 'haar')
#print(approx_coeffs_abs,coeffs[1])
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(noisy_image, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(reconstructed, cmap='bwr')
ax[1].set_title('UWT Thresholded and Reconstructed')
plt.show()