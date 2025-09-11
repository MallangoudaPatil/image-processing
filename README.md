# Image Processing

This folder contains fundamental **image processing techniques** implemented using **OpenCV** and **Python**.  
Each script demonstrates a specific operation, visualizes the results, and saves the output for easy comparison.

---

## Techniques Implemented

### 1. Grayscale Conversion
Converts a color image into grayscale format (intensity-based representation).  

**Result:**  
<img width="800" height="300" alt="Grayscale Conversion" src="https://github.com/user-attachments/assets/89c4019a-3e75-4c39-b890-fe637e68befb" />

---

### 2. Histogram Equalization
Enhances image contrast by spreading out the intensity values across the histogram.  

**Result:**  
<img width="800" height="500" alt="Histogram Equalization" src="https://github.com/user-attachments/assets/da6d367b-5ce3-4678-8d00-d251e08c9a40" />

---

### 3. Edge Detection
Detects object boundaries using multiple operators: **Sobel**, **Laplacian**, and **Canny**.  

**Result:**  
<img width="800" height="500" alt="Edge Detection" src="https://github.com/user-attachments/assets/e6d46576-857b-4873-af88-aef6fbc74bc1" />

---

### 4. Thresholding
Segments an image into foreground and background using:  
- Simple Thresholding  
- Adaptive Thresholding  
- Otsu’s Thresholding  

*(Add screenshot here once result is generated)*  

---

### 5. Image Blurring
Smooths images and reduces noise using different filters:  
- Average Blurring  
- Gaussian Blurring  
- Median Blurring  
- Bilateral Filtering  

**Result:**  
<img width="800" height="500" alt="Image Blurring" src="https://github.com/user-attachments/assets/f5133b25-c0ed-4f9d-a7ca-83e2970d4221" />

---

### 6. Resize & Rotate
Applies basic geometric transformations such as **resizing** (scale up/down) and **rotation**.  

**Result:**  
<img width="800" height="500" alt="Resize and Rotate" src="https://github.com/user-attachments/assets/f5dc77a2-86ae-4192-9a63-e818359198cd" />

---

## ⚙️ Setup Instructions

Install the required dependencies:

```bash
pip install opencv-python numpy matplotlib
