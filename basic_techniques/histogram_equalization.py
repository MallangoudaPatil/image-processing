import cv2
import os
import matplotlib.pyplot as plt

# Read the image in grayscale mode
image = cv2.imread("D:\\GitHub\\image-processing\\test-images-for-Image-Processing\\fruits.png", cv2.IMREAD_GRAYSCALE)

# Apply Histogram Equalization
equalized = cv2.equalizeHist(image)

# Plot before and after
plt.figure(figsize=(10, 6))

# Original image
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Equalized image
plt.subplot(2, 2, 2)
plt.imshow(equalized, cmap='gray')
plt.title("Equalized Image")
plt.axis("off")

# Histogram of original image
plt.subplot(2, 2, 3)
plt.hist(image.ravel(), bins=256, range=[0, 256], color='blue')
plt.title("Original Histogram")

# Histogram of equalized image
plt.subplot(2, 2, 4)
plt.hist(equalized.ravel(), bins=256, range=[0, 256], color='green')
plt.title("Equalized Histogram")

plt.tight_layout()
plt.show()