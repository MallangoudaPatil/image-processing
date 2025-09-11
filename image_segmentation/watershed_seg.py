import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image
img = cv2.imread("D:\\GitHub\\image-processing\\test-images-for-Image-Processing\\peppers.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Apply threshold
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 2: Remove noise using Morphology
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 3: Sure background area (Dilate)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Step 4: Sure foreground area (Distance transform + threshold)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Step 5: Unknown region (Subtract FG from BG)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Step 6: Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add 1 to all labels so that sure background is not 0
markers = markers + 1

# Mark the unknown region with 0
markers[unknown == 255] = 0

# Step 7: Apply Watershed
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]  # Mark boundaries with red

# Display results
plt.figure(figsize=(15,6))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Watershed Segmentation Result")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(thresh, cmap="gray")
plt.title("Binary Image (Otsu Threshold)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(markers, cmap="jet")
plt.title("Marker Labels")
plt.axis("off")

plt.show()
