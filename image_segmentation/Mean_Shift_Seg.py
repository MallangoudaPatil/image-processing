import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image
img = cv2.imread("D:\\GitHub\\image-processing\\test-images-for-Image-Processing\\peppers.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply Mean Shift Filtering
# Parameters: (image, spatial_radius, color_radius)
# - spatial_radius: affects neighborhood size
# - color_radius: affects color similarity
segmented = cv2.pyrMeanShiftFiltering(img, sp=20, sr=40)

# Convert result to RGB
segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)

# Plot results
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(segmented_rgb)
plt.title("Mean Shift Segmentation")
plt.axis("off")

plt.show()
