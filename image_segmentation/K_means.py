import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image
img = cv2.imread("D:\\GitHub\\image-processing\\test-images-for-Image-Processing\\peppers.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR -> RGB

# Reshape the image into a 2D array of pixels (rows: pixels, cols: RGB channels)
pixels = img.reshape((-1, 3))
pixels = np.float32(pixels)

# Define criteria, clusters (K) and apply KMeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 2  # Number of clusters
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert back to 8-bit values
centers = np.uint8(centers)

# Map each pixel to the centroid color
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(img.shape)

# Plot original vs segmented
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(segmented_image)
plt.title(f"K-Means Segmentation (k={k})")
plt.axis("off")

plt.show()
