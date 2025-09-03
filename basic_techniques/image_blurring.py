import cv2
import matplotlib.pyplot as plt

# Read image (in color for better visualization)
image = cv2.imread("D:\\GitHub\\image-processing\\test-images-for-Image-Processing\\tulips.png")

# 1. Average Blurring
avg_blur = cv2.blur(image, (5, 5))

# 2. Gaussian Blurring
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# 3. Median Blurring
median_blur = cv2.medianBlur(image, 5)

# 4. Bilateral Filtering (edge-preserving)
bilateral_blur = cv2.bilateralFilter(image, 9, 75, 75)

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(avg_blur, cv2.COLOR_BGR2RGB))
plt.title("Average Blurring")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
plt.title("Gaussian Blurring")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB))
plt.title("Median Blurring")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(bilateral_blur, cv2.COLOR_BGR2RGB))
plt.title("Bilateral Filtering")
plt.axis("off")

plt.tight_layout()
plt.show()
