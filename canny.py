import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import data

img = data.camera()

#canny
img_canny = cv2.Canny(img,100,200)

fig, axes = plt.subplots(2,2, figsize=(10,10))
ax = axes.ravel()

ax[0].imshow(img, cmap = 'gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins = 256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_canny, cmap = 'gray')
ax[2].set_title("Citra Input")
ax[3].hist(img_canny.ravel(), bins = 256)
ax[3].set_title("Histogram Citra Output")

fig.tight_layout()

plt.show()