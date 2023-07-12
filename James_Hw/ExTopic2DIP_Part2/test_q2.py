import numpy as np
from numpy import fft
import scipy
from scipy import misc, fftpack
import cv2
from matplotlib import pyplot as plt
# def realget(img):
#     realpt = img[:,:,0]
#     imagpt = img[:,:,1]
#     np.add(realpt,np.multiply(imagpt,1.j))
#     np.
input_imageThumbPrint = cv2.imread('ThumbPrint.tif')
input_imageFlorida = cv2.imread('Florida.tif')
input_imageMountainHalftone = cv2.imread('MountainHalftone.png')
# img = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
mid_thumbprint = cv2.dft(np.float32(cv2.cvtColor(np.float32(input_imageThumbPrint), cv2.COLOR_BGR2GRAY)), flags=cv2.DFT_COMPLEX_OUTPUT)
mid_florida = cv2.dft(np.float32(cv2.cvtColor(np.float32(input_imageFlorida), cv2.COLOR_BGR2GRAY)), flags=cv2.DFT_COMPLEX_OUTPUT)
mid_MountainHalftone = cv2.dft(np.float32(cv2.cvtColor(np.float32(input_imageMountainHalftone), cv2.COLOR_BGR2GRAY)), flags=cv2.DFT_COMPLEX_OUTPUT)
mid_thumbprint=np.fft.fftshift(mid_thumbprint, axes=(1,))
mid_florida=np.fft.fftshift(mid_florida, axes=(1,))
mid_MountainHalftone=np.fft.fftshift(mid_MountainHalftone, axes=(1,))
end_ThumbPrint = cv2.GaussianBlur(input_imageThumbPrint, ksize=(5, 5), sigmaX=2, borderType=cv2.BORDER_REPLICATE)
end_Florida = cv2.GaussianBlur(input_imageFlorida, ksize=(5, 5), sigmaX=2, borderType=cv2.BORDER_REPLICATE)
end_MountainHalftone = cv2.GaussianBlur(input_imageMountainHalftone, ksize=(5, 5), sigmaX=2, borderType=cv2.BORDER_REPLICATE)

plt.subplot(2,3,1)
plt.imshow(end_ThumbPrint)
plt.subplot(2,3,2)
plt.imshow(end_Florida)
plt.subplot(2,3,3)
plt.imshow(end_MountainHalftone)
plt.subplot(2,3,4)
plt.imshow(mid_thumbprint[:,:,0].astype("uint8"))
plt.subplot(2,3,5)
plt.imshow(mid_florida[:,:,0].astype("uint8"))
plt.subplot(2,3,6)
plt.imshow(mid_MountainHalftone[:,:,0].astype("uint8"))
# plt.subplot(2,3,4)
# plt.imshow(np.real(mid_thumbprint))
# plt.subplot(2,3,5)
# plt.imshow(np.real(mid_florida))
# plt.subplot(2,3,6)
# plt.imshow(np.real(mid_MountainHalftone))
plt.show()