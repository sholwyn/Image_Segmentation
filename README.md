
# Image Segmentation Project

## Overview

This project implements image segmentation with a focus on detecting **hemorrhagic regions** using **K-means clustering** and **edge detection** techniques. The system processes **medical images** to segment hemorrhagic areas, visualize the results, and evaluate the segmentation using performance metrics like **PSNR**, **SSIM**, **mIoU**, and more.

The main techniques used in this project include:

* **Preprocessing** with **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and **bilateral filtering**.
* **Edge Detection** using **Sobel** and **Canny** methods.
* **Feature Extraction** using **Local Binary Pattern (LBP)**.
* **Segmentation** using **K-means clustering**.
* **Performance Metrics** evaluation such as **SSIM**, **PSNR**, **mIoU**, **Dice Coefficient**, **Sensitivity**, **Specificity**, and **Hausdorff Distance**.

## Features

* Preprocesses medical images for enhanced feature extraction.
* Detects edges and segments hemorrhagic regions in medical images.
* Evaluates segmentation quality using various metrics.
* Visualizes results in both **2D** and **3D** formats.
* Supports **real-time segmentation** for input images.

## Requirements

To run this project, install the following dependencies:

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
