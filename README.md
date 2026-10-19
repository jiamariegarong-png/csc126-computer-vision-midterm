# Computer Vision Security & Motion Monitor

## Project Overview
This project is developed for the **CSC 126 Computer Vision Midterm Project** at Caraga State University (CCIS)[cite: 1, 2]. It is a real-time security and motion-monitoring application built using Python and OpenCV[cite: 1, 2].

## Integrated OpenCV Techniques
1. **Image Operations:** Utilizes grayscale conversion, Gaussian blurring, resizing, and absolute difference calculations[cite: 2].
2. **Image Segmentation & Contours:** Applies thresholding and contour extraction to isolate and bound moving objects in the live feed[cite: 2, 4].
3. **Haar Cascade Classifier:** Detects human faces in real-time alongside motion tracking[cite: 2].

## Setup Instructions
1. Ensure you have **Python** installed on your system.
2. Clone or download this repository into your workspace.
3. Open the project folder in **VS Code**.
4. Install the required dependencies by opening your terminal in VS Code and running:
   ```bash
   pip install -r requirements.txt