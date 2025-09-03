import cv2
import os
import matplotlib.pyplot as plt

# Read image in grayscale
image = cv2.imread("D:\\GitHub\\image-processing\\test-images-for-Image-Processing\\peppers.png", cv2.IMREAD_GRAYSCALE)

# Canny Edge Detection
canny_edges = cv2.Canny(image, 100, 200)

# 2. Sobel Edge Detection (X and Y)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# 3. Laplacian Edge Detection
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(canny_edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(sobel_combined, cmap="gray")
plt.title("Sobel Edge Detection")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.show()
