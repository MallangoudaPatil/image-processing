import cv2
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("D:\\GitHub\\image-processing\\test-images-for-Image-Processing\\peppers.png", 0)

# Global thresholding (manual)
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu's thresholding
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Otsu’s thresholding after Gaussian filtering (better for noisy images)
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Plot results
plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(th1, cmap="gray")
plt.title("Global Thresholding (t=127)")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(th2, cmap="gray")
plt.title(f"Otsu Thresholding (t={ret2:.2f})")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(th3, cmap="gray")
plt.title(f"Gaussian + Otsu (t={ret3:.2f})")
plt.axis("off")

plt.show()
