import pywt
import numpy as np
from matplotlib import pyplot as plt
import cv2

# Python3 code to demonstrate working of
# Flatten tuple of List to tuple
# Using tuple() + chain.from_iterable()
from itertools import chain
# Define the input signal
x = cv2.imread('lena512x512.png')
# x = np.random.randn(1024)

# Apply the DWT
coeffs = pywt.wavedec2(x, 'db4', level=5)

# Sort the coefficients
sorted_coeffs = np.sort(np.abs(np.concatenate(coeffs)))

# Keep only the top 10% of coefficients
keep = int(np.ceil(len(sorted_coeffs) * 0.1))
threshold = sorted_coeffs[-keep]
modified_coeffs = [c if abs(c) >= threshold else 0 for c in coeffs]

# Reconstruct the signal using the modified coefficients
x_compressed = pywt.waverec2(modified_coeffs, 'db4')

# Calculate the compression ratio
compression_ratio = len(x) / len(modified_coeffs[0])

# import pywt
# import numpy as np
# from PIL import Image

# # Load the image
# image = Image.open('lena512x512.png')

# # Convert the image to grayscale
# image = image.convert('L')

# # Convert the image to a numpy array
# x = np.array(image)

# # Apply the DWT
# coeffs = pywt.wavedec2(x, 'db4', level=5)

# # Sort the coefficients
# def recursion_Flat(item):
#     arr = []
#     for i in item:
#         if type(i) == list or type(i)==tuple or isinstance(i, np.ndarray):
#             nu = recursion_Flat(i)
#             arr.extend(nu)
#             # for j in i:
#             #     nu = recursion_Flat(j)
#         else:
#             arr.append(i)
#         # elif type(i) == i
#     return arr
# # sorted_coeffs = np.sort(np.abs(np.concatenate([list(recursion_Flat(c))  for c in coeffs if np.size(c)> 0])))
# # sorted_coeffs = np.sort(np.abs(np.concatenate([list(chain.from_iterable(c))  for c in coeffs if np.size(c)> 0])))
# # sorted_coeffs = np.sort(np.abs(np.concatenate([c.ravel() for c in coeffs])))
# sorted_coeffs = np.sort(np.abs(np.concatenate([np.ravel(c) for c in coeffs])))

# # Keep only the top 10% of coefficients
# keep = int(np.ceil(len(sorted_coeffs) * 0.1))
# threshold = sorted_coeffs[-keep]
# modified_coeffs = [pywt.threshold(c, threshold) for c in coeffs]

# # Reconstruct the image using the modified coefficients
# x_compressed = pywt.waverec2(modified_coeffs, 'db4')

# # Convert the numpy array to an image
# image_compressed = Image.fromarray(np.uint8(x_compressed))

# # Save the compressed image
# # image_compressed.save('image_compressed.jpg')


plt.imshow(x_compressed)
# plt.imshow(image_compressed)
# plt.imshow(x_compressed)
plt.show()