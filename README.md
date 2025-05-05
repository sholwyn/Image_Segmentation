# Image Segmentation Project

## Overview
This project implements image segmentation with a focus on detecting brain hemorrhages using K-means clustering and edge detection. The project processes medical images (specifically brain hemorrhage images) to segment hemorrhagic regions, visualize the results, and evaluate the segmentation using performance metrics like PSNR, SSIM, mIoU, and more.

The main techniques used in this project include:
- Preprocessing with CLAHE (Contrast Limited Adaptive Histogram Equalization) and bilateral filtering.
- Edge Detection using Sobel and Canny methods.
- Feature Extraction using Local Binary Pattern (LBP).
- Segmentation using K-means clustering.
- Performance Metrics evaluation such as SSIM, PSNR, mIoU, Dice Coefficient, Sensitivity, Specificity, and Hausdorff Distance.

## Features
- Preprocesses input images for better feature extraction.
- Detects edges and segments hemorrhagic regions in medical images.
- Evaluates segmentation performance using multiple metrics.
- Visualizes results in both 2D and 3D.
- Provides real-time segmentation for input images.

## Requirements
To run this project, you need to install the following dependencies:

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- scikit-learn
- pandas
- scipy

You can install the dependencies by running:

```bash
pip install -r requirements.txt
