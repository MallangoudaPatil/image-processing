import cv2
import os

# image
image = cv2.imread("D:\\GitHub\\image-processing\\test-images-for-Image-Processing\\fruits.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display original and grayscale images
cv2.imshow("Original Image", image)
cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
