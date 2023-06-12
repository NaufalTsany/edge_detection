import cv2
from matplotlib import pyplot as plt

# Membaca Gambar
img0 = cv2.imread('images/freya.jfif')

# Konversi ke gray
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# Hilangkan noise
img = cv2.GaussianBlur(gray, (3, 3), 0)

# Konvolusi dengan kernel
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Tampilkan dengan matplotlib
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian')
plt.xticks([])
plt.yticks([])

plt.show()