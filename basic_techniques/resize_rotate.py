import cv2
import os
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("D:\\GitHub\\image-processing\\test-images-for-Image-Processing\\tulips.png")

h, w = image.shape[:2]

# 1. Resizing
resized_half = cv2.resize(image, (w // 2, h // 2))
resized_double = cv2.resize(image, (w * 2, h * 2))

# 2. Rotation (90°, 180°, 270°)
rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(resized_half, cv2.COLOR_BGR2RGB))
plt.title("Resized (Half)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(resized_double, cv2.COLOR_BGR2RGB))
plt.title("Resized (Double)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(rotated_90, cv2.COLOR_BGR2RGB))
plt.title("Rotated 90°")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(rotated_180, cv2.COLOR_BGR2RGB))
plt.title("Rotated 180°")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(rotated_270, cv2.COLOR_BGR2RGB))
plt.title("Rotated 270°")
plt.axis("off")

plt.tight_layout()
plt.show()
