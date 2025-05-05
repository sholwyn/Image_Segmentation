
# Image Segmentation Project
> Team Members:
Name	Roll Number	Email
Sholwyn 4SO22CD050
Sruthi  4SO22CD053
Yash    4SO22CD063

> Problem Statement:
The goal of this project is to automatically detect and segment brain tumors from MRI scans using digital image processing (DIP) techniques and machine learning-based segmentation methods. The aim is to improve the visibility of tumors for diagnosis and treatment planning.

> Techniques & Program Flow:
## Original Image Loading:

Input: MRI brain scan images (grayscale or color).

Purpose: Serve as the baseline for all processing steps.

## CLAHE Enhancement:

Technique: Contrast Limited Adaptive Histogram Equalization.

Purpose: Enhances local contrast to make tumor boundaries clearer.

## Edge Detection:

Technique: Canny or Sobel Edge Detector.

Purpose: Detects the edges within the brain to highlight boundaries of tissues and tumors.

## HOG Feature Extraction:

Technique: Histogram of Oriented Gradients.

Purpose: Captures texture and gradient features helpful in identifying structural differences.

## K-means Clustering:

Technique: Unsupervised clustering (k=2 or 3).

Purpose: Segments the image into different clusters (tumor vs. non-tumor).

## Random Forest Segmentation:

Technique: Supervised machine learning (Random Forest Classifier).

Purpose: Learns from labeled data to improve segmentation accuracy.

## Watershed Segmentation:

Technique: Watershed Algorithm.

Purpose: Refines object boundaries by detecting contours and segmenting overlapping regions.

## Final Segmentation Mask:

Combination: The best segmentation outputs are combined into a binary mask for tumor regions.

## Overlay and Visualization:

Technique: Color mask overlay (red/green) using OpenCV or Matplotlib.

Purpose: Highlights tumor regions on top of the original MRI scan for clear visualization.
* Python 3.x
* OpenCV
* NumPy
* Matplotlib
* scikit-image
* scikit-learn
* pandas
* scipy

Install all dependencies using:

```bash
pip install -r requirements.txt
```
