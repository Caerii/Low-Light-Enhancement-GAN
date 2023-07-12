import numpy as np
import PIL
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage.exposure
def stretch_image(img, low_thresh, high_thresh):
    # Compute the percentiles of the image
    p_low = np.percentile(img, low_thresh)
    p_high = np.percentile(img, high_thresh)

    # Stretch the image between the low and high percentiles
    stretched = (img - p_low) / (p_high - p_low)
    stretched[stretched < 0] = 0
    stretched[stretched > 1] = 1

    return stretched
# Load the first image as a PIL image object and convert to a NumPy array
img = PIL.Image.open('C:\\Users\\Josepl\\OneDrive\\Desktop\\DTU course\\Tpc2Part1\\Tpc2Part1\\Pics\\goldhill.png')
img_array = np.asarray(img)

# Display the first image
plt.imshow(img)
plt.title('Original Image')
plt.show()


# Load the image as a PIL image object
img = PIL.Image.open('C:\\Users\\Josep\\OneDrive\\Desktop\\DTU course\\Tpc2Part1\\Tpc2Part1\\Pics\\PVPanel_electroluminescence.tif')

# Convert the PIL image to a NumPy array
img_array = np.asarray(img)

# Calculate the minimum and maximum pixel values
img_min = np.min(img_array)
img_max = np.max(img_array)

# Normalize the image values to the range 0-1
normalized = (img_array - img_min) / (img_max - img_min)

# Display the normalized image
plt.imshow(normalized, cmap='gist_gray')
plt.title('Normalized Image')
plt.show()

# Load the image as a NumPy array
img = plt.imread('C:\\Users\\Josep\\OneDrive\\Desktop\\DTU course\\Tpc2Part1\\Tpc2Part1\\Pics\\goldhill.png')

# Create a 3D plot of the image as a wireframe mesh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
ax.plot_surface(x, y, img, cmap='gray', linewidth=0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Pixel Value')
plt.title('Image as a Wireframe Mesh')
plt.show()
img = plt.imread('C:\\Users\\Josep\\OneDrive\\Desktop\\DTU course\\Tpc2Part1\\Tpc2Part1\\Pics\\NYCSunset.jpg')
img_array = np.asarray(img)
historical = skimage.exposure.equalize_hist(img_array, nbins=256, mask=None)

plt.imshow(historical)
plt.show()
img = plt.imread('C:\\Users\\Josep\\OneDrive\\Desktop\\DTU course\\Tpc2Part1\\Tpc2Part1\\Pics\\goldhill.png')

# Stretch the image between the 0.1 and 0.9 percentiles
stretched = stretch_image(img, 0.1, 99.0)

# Plot the histograms of the original and modified images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].hist(img.ravel(), bins=256, range=(0, 1))
axs[0].set_title('Original Image')
axs[1].hist(stretched.ravel(), bins=256, range=(0, 1))
axs[1].set_title('Stretched Image')
plt.show()
plt.imshow(stretched)
plt.show()

img = plt.imread('C:\\Users\\Josep\\OneDrive\\Desktop\\DTU course\\Tpc2Part1\\Tpc2Part1\\Pics\\PamplonaSunset.jpg')
img_array = np.asarray(img)
historical = skimage.exposure.equalize_hist(img_array, nbins=256, mask=None)

plt.imshow(historical)
plt.show()