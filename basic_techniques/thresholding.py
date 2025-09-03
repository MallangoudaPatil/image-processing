import cv2
import matplotlib.pyplot as plt

# Read image in grayscale
image = cv2.imread("D:\\GitHub\\image-processing\\test-images-for-Image-Processing\\tulips.png", cv2.IMREAD_GRAYSCALE)

# 1. Simple Thresholding
_, simple_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 2. Adaptive Mean Thresholding
adaptive_mean = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)

# 3. Adaptive Gaussian Thresholding
adaptive_gaussian = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# 4. Otsu’s Thresholding
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
