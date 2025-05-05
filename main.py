#!/usr/bin/env python3
# Advanced Brain Hemorrhage Segmentation System
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
from skimage import exposure
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import os
import time
import pandas as pd
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap


class HemorrhageSegmentation:
    def __init__(self, image_path):
        """Initialize the hemorrhage segmentation system."""
        self.image_path = image_path
        self.original_image = None
        self.gray_image = None
        self.segmented_image = None
        self.segmentation_mask = None
        self.metrics = {}
        self.load_image()

    def load_image(self):
        """Load and prepare the image."""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Image not found at path: {self.image_path}")

        # Create a copy for results
        self.result_image = self.original_image.copy()

        # Convert to grayscale
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Store image dimensions
        self.height, self.width = self.gray_image.shape[:2]
        print(f"Image loaded successfully: {self.width}x{self.height}")

    def preprocess(self):
        """Advanced preprocessing pipeline."""
        print("Preprocessing image...")

        # Histogram equalization for better contrast
        self.equalized = cv2.equalizeHist(self.gray_image)

        # Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.clahe_image = clahe.apply(self.gray_image)

        # Bilateral filter - edge-preserving smoothing
        self.bilateral = cv2.bilateralFilter(self.clahe_image, d=9, sigmaColor=90, sigmaSpace=85)

        # Advanced edge detection using a combination of gradients
        sobelx = cv2.Sobel(self.bilateral, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.bilateral, cv2.CV_64F, 0, 1, ksize=3)
        self.gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2).astype(np.uint8)

        # Enhanced Canny edge detection
        self.edges = cv2.Canny(self.bilateral, 40, 180)

        # Get noise-reduced image for segmentation
        self.preprocessed = self.bilateral.copy()
        return self.preprocessed

    def extract_features(self):
        """Extract advanced image features."""
        print("Extracting image features...")

        # HOG features
        fd, self.hog_image = hog(
            self.clahe_image,
            orientations=8,
            pixels_per_cell=(32, 32),
            cells_per_block=(1, 1),
            visualize=True,
            block_norm='L2-Hys'
        )
        self.hog_image = exposure.rescale_intensity(self.hog_image, in_range=(0, 10))
        self.hog_image = (self.hog_image * 255).astype(np.uint8)

        # Generate texture features using LBP (Local Binary Pattern)
        from skimage.feature import local_binary_pattern
        radius = 3
        n_points = 8 * radius
        self.lbp = local_binary_pattern(self.gray_image, n_points, radius, method='uniform')

        # Create feature vector
        self.feature_image = np.dstack([
            self.clahe_image,
            self.gradient_magnitude,
            self.hog_image,
            self.lbp.astype(np.uint8)
        ])

        return self.feature_image

    def segment_kmeans(self, n_clusters=3):
        """Segment using K-means clustering on feature space."""
        print(f"Performing K-means segmentation with {n_clusters} clusters...")

        # Reshape data for k-means
        feature_data = self.feature_image.reshape((-1, self.feature_image.shape[2]))

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(feature_data)

        # Reshape result back to image dimensions
        segmented = kmeans.labels_.reshape(self.height, self.width)

        # Find the cluster that likely represents the hemorrhage
        # (typically the brightest areas in the original image)
        mean_intensities = []
        for i in range(n_clusters):
            mask = (segmented == i).astype(np.uint8)
            mean_intensity = np.mean(self.clahe_image[mask == 1])
            mean_intensities.append((i, mean_intensity))

        # Sort clusters by intensity, brightest cluster is likely hemorrhage
        mean_intensities.sort(key=lambda x: x[1], reverse=True)
        hemorrhage_cluster = mean_intensities[0][0]

        # Create mask for the hemorrhage cluster
        self.segmentation_mask = (segmented == hemorrhage_cluster).astype(np.uint8) * 255

        # Post-process the mask
        kernel = np.ones((5, 5), np.uint8)
        self.segmentation_mask = cv2.morphologyEx(self.segmentation_mask, cv2.MORPH_OPEN, kernel)
        self.segmentation_mask = cv2.morphologyEx(self.segmentation_mask, cv2.MORPH_CLOSE, kernel)

        return self.segmentation_mask

    def segment_watershed(self):
        """Advanced watershed segmentation."""
        print("Performing watershed segmentation...")

        # Use optimized thresholding
        ret, thresh = cv2.threshold(self.bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        # Apply watershed algorithm
        markers = cv2.watershed(self.original_image, markers)

        # Create mask where the boundaries are marked
        self.watershed_mask = np.zeros_like(self.gray_image, dtype=np.uint8)
        self.watershed_mask[markers == -1] = 255

        # Combine with K-means mask for better results
        self.final_mask = cv2.bitwise_or(self.segmentation_mask, self.watershed_mask)

        # Apply the mask to the original image
        self.segmented_image = self.original_image.copy()
        self.segmented_image[markers == -1] = [0, 0, 255]  # Mark boundaries in red

        # Create overlay for hemorrhage areas
        hemorrhage_overlay = np.zeros_like(self.original_image)
        hemorrhage_area = (self.segmentation_mask > 0)
        hemorrhage_overlay[hemorrhage_area] = [0, 255, 0]  # Green for hemorrhage

        # Blend with original image
        alpha = 0.4
        self.result_image = cv2.addWeighted(self.original_image, 1, hemorrhage_overlay, alpha, 0)

        # Add watershed boundaries
        self.result_image[markers == -1] = [255, 0, 0]  # Red for boundaries

        return self.result_image

    def random_forest_segmentation(self, sample_fraction=0.1):
        """Use Random Forest classifier for segmentation."""
        print("Performing Random Forest segmentation...")

        # Sample pixels for training
        n_samples = int(self.height * self.width * sample_fraction)

        # Get indices for random samples
        indices = np.random.choice(self.height * self.width, n_samples, replace=False)
        rows = indices // self.width
        cols = indices % self.width

        # Create initial segmentation based on intensity thresholding for training
        _, binary = cv2.threshold(self.clahe_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_flat = binary.flatten()

        # Create feature matrix for sampled pixels
        features = np.zeros((n_samples, 4))
        for i, (r, c) in enumerate(zip(rows, cols)):
            features[i, 0] = self.clahe_image[r, c]
            features[i, 1] = self.gradient_magnitude[r, c]
            features[i, 2] = self.hog_image[r, c]
            features[i, 3] = self.lbp[r, c]

        # Extract labels for sampled pixels
        labels = binary_flat[indices] // 255

        # Train Random Forest classifier
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        rf.fit(features, labels)

        # Now predict for all pixels
        all_features = np.zeros((self.height * self.width, 4))
        for i in range(self.height):
            for j in range(self.width):
                idx = i * self.width + j
                all_features[idx, 0] = self.clahe_image[i, j]
                all_features[idx, 1] = self.gradient_magnitude[i, j]
                all_features[idx, 2] = self.hog_image[i, j]
                all_features[idx, 3] = self.lbp[i, j]

        # Make predictions
        predictions = rf.predict(all_features)

        # Reshape predictions to image format
        self.rf_mask = predictions.reshape(self.height, self.width) * 255
        self.rf_mask = self.rf_mask.astype(np.uint8)

        # Post-process the random forest mask
        kernel = np.ones((5, 5), np.uint8)
        self.rf_mask = cv2.morphologyEx(self.rf_mask, cv2.MORPH_OPEN, kernel)
        self.rf_mask = cv2.morphologyEx(self.rf_mask, cv2.MORPH_CLOSE, kernel)

        # Combine with the other masks
        self.final_mask = cv2.bitwise_or(self.final_mask, self.rf_mask)

        return self.rf_mask

    def calculate_metrics(self):
        """Calculate comprehensive segmentation metrics."""
        print("Calculating performance metrics...")

        # Create a binary segmented image
        segmented_binary = (self.final_mask > 0).astype(np.uint8) * 255

        # Convert original to binary for comparison
        _, original_binary = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculate SSIM
        self.metrics['ssim'], _ = ssim(original_binary, segmented_binary, full=True)

        # Calculate PSNR
        self.metrics['psnr'] = cv2.PSNR(original_binary, segmented_binary)

        # Calculate IoU (Intersection over Union)
        intersection = np.logical_and(original_binary, segmented_binary).sum()
        union = np.logical_or(original_binary, segmented_binary).sum()
        self.metrics['iou'] = intersection / union if union > 0 else 0

        # Calculate Dice coefficient
        dice = 2 * intersection / (original_binary.sum() + segmented_binary.sum()) if (
                                                                                                  original_binary.sum() + segmented_binary.sum()) > 0 else 0
        self.metrics['dice'] = dice

        # Calculate sensitivity and specificity
        # True positive: hemorrhage pixels correctly identified
        tp = np.logical_and(original_binary > 0, segmented_binary > 0).sum()
        # True negative: non-hemorrhage pixels correctly identified
        tn = np.logical_and(original_binary == 0, segmented_binary == 0).sum()
        # False positive: non-hemorrhage pixels incorrectly identified as hemorrhage
        fp = np.logical_and(original_binary == 0, segmented_binary > 0).sum()
        # False negative: hemorrhage pixels incorrectly identified as non-hemorrhage
        fn = np.logical_and(original_binary > 0, segmented_binary == 0).sum()

        self.metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        self.metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

        # Calculate Hausdorff distance (boundary accuracy)
        from scipy.spatial.distance import directed_hausdorff

        # Get contours
        contours_orig, _ = cv2.findContours(original_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_seg, _ = cv2.findContours(segmented_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_orig and contours_seg:
            # Get the largest contour from each
            largest_orig = max(contours_orig, key=cv2.contourArea)
            largest_seg = max(contours_seg, key=cv2.contourArea)

            # Convert contours to point arrays
            points_orig = largest_orig.reshape(-1, 2)
            points_seg = largest_seg.reshape(-1, 2)

            # Calculate Hausdorff distance
            if len(points_orig) > 0 and len(points_seg) > 0:
                d1 = directed_hausdorff(points_orig, points_seg)[0]
                d2 = directed_hausdorff(points_seg, points_orig)[0]
                self.metrics['hausdorff'] = max(d1, d2)
            else:
                self.metrics['hausdorff'] = float('inf')
        else:
            self.metrics['hausdorff'] = float('inf')

        # Print metrics
        print("\n========= PERFORMANCE METRICS =========")
        for key, value in self.metrics.items():
            print(f"{key.upper()} : {value:.4f}")
        print("=======================================\n")

        return self.metrics

    def visualize_results(self):
        """Create comprehensive visualization of results."""
        print("Generating visualizations...")

        # Set up the figure with 3 rows and 3 columns
        plt.figure(figsize=(16, 12))

        # Original image
        plt.subplot(3, 3, 1)
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        # Preprocessed image
        plt.subplot(3, 3, 2)
        plt.imshow(self.clahe_image, cmap='gray')
        plt.title("CLAHE Enhanced")
        plt.axis("off")

        # Edge detection
        plt.subplot(3, 3, 3)
        plt.imshow(self.edges, cmap='gray')
        plt.title("Edge Detection")
        plt.axis("off")

        # HOG Features
        plt.subplot(3, 3, 4)
        plt.imshow(self.hog_image, cmap='gray')
        plt.title("HOG Features")
        plt.axis("off")

        # K-means segmentation
        plt.subplot(3, 3, 5)
        plt.imshow(self.segmentation_mask, cmap='gray')
        plt.title("K-means Segmentation")
        plt.axis("off")

        # Random Forest segmentation
        plt.subplot(3, 3, 6)
        plt.imshow(self.rf_mask, cmap='gray')
        plt.title("Random Forest Segmentation")
        plt.axis("off")

        # Watershed segmentation
        plt.subplot(3, 3, 7)
        plt.imshow(self.watershed_mask, cmap='gray')
        plt.title("Watershed Boundaries")
        plt.axis("off")

        # Final segmentation mask
        plt.subplot(3, 3, 8)
        plt.imshow(self.final_mask, cmap='gray')
        plt.title("Final Segmentation Mask")
        plt.axis("off")

        # Final result overlaid on original image
        plt.subplot(3, 3, 9)
        plt.imshow(cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB))
        plt.title("Final Result")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig("hemorrhage_segmentation_results.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Create 3D visualization of the segmentation
        self.create_3d_visualization()

        # Generate heatmap visualization
        self.create_heatmap_visualization()

    def create_3d_visualization(self):
        """Create 3D visualization of the segmentation."""
        from mpl_toolkits.mplot3d import Axes3D

        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create x, y coordinates
        x, y = np.meshgrid(np.arange(0, self.width, 4), np.arange(0, self.height, 4))

        # Downsample for performance
        z = self.clahe_image[::4, ::4]
        mask = self.final_mask[::4, ::4]

        # Create color map for hemorrhage areas
        colors = np.zeros((*mask.shape, 4))  # RGBA
        colors[mask > 0] = [1, 0, 0, 0.7]  # Red for hemorrhage
        colors[mask == 0] = [0.5, 0.5, 0.5, 0.2]  # Gray for background

        # Plot the surface
        surf = ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1,
                               linewidth=0, antialiased=True)

        ax.set_title('3D Visualization of Brain Hemorrhage')
        ax.set_zlim(0, 255)

        plt.savefig("hemorrhage_3d_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()

    def create_heatmap_visualization(self):
        """Create heatmap visualization of probability distribution."""
        # Use distance transform as a probability measure
        dist = ndimage.distance_transform_edt(self.final_mask)
        dist = dist / dist.max()  # Normalize to [0, 1]

        # Create a custom colormap: transparent to red
        colors = [(0, 0, 0, 0), (1, 0, 0, 1)]
        cmap = LinearSegmentedColormap.from_list('hemorrhage_cmap', colors, N=256)

        plt.figure(figsize=(12, 6))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        # Heatmap overlay
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.imshow(dist, cmap=cmap, alpha=0.7)
        plt.colorbar(label='Hemorrhage Probability')
        plt.title("Hemorrhage Probability Heatmap")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig("hemorrhage_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self):
        """Generate a comprehensive report of the analysis."""
        # Create a dataframe for the metrics
        metrics_df = pd.DataFrame([self.metrics])

        print("\n=============== HEMORRHAGE ANALYSIS REPORT ===============")
        print(f"Image: {self.image_path}")
        print(f"Dimensions: {self.width}x{self.height}")

        # Calculate hemorrhage statistics
        hemorrhage_pixels = np.sum(self.final_mask > 0)
        total_pixels = self.width * self.height
        hemorrhage_percentage = (hemorrhage_pixels / total_pixels) * 100

        print(f"Hemorrhage area: {hemorrhage_pixels} pixels ({hemorrhage_percentage:.2f}% of brain)")
        print("\nPerformance Metrics:")
        print(metrics_df.to_string(index=False))

        # Generate HTML report
        html_report = f"""
        <html>
        <head>
            <title>Brain Hemorrhage Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; }}
                .metrics {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                .metrics td, .metrics th {{ border: 1px solid #ddd; padding: 8px; }}
                .metrics tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metrics th {{ padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #3498db; color: white; }}
                .image-container {{ display: flex; justify-content: center; margin: 20px 0; }}
                .image {{ margin: 0 10px; text-align: center; }}
                .caption {{ margin-top: 5px; font-style: italic; }}
            </style>
        </head>
        <body>
            <h1>Brain Hemorrhage Analysis Report</h1>
            <p><strong>Image:</strong> {self.image_path}</p>
            <p><strong>Dimensions:</strong> {self.width}x{self.height}</p>
            <p><strong>Analysis Date:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Hemorrhage Statistics</h2>
            <p>Hemorrhage area: {hemorrhage_pixels} pixels ({hemorrhage_percentage:.2f}% of brain)</p>

            <h2>Performance Metrics</h2>
            <table class="metrics">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """

        for key, value in self.metrics.items():
            html_report += f"""
                <tr>
                    <td>{key.upper()}</td>
                    <td>{value:.4f}</td>
                </tr>
            """

        html_report += """
            </table>

            <h2>Visualizations</h2>
            <div class="image-container">
                <div class="image">
                    <img src="hemorrhage_segmentation_results.png" width="600">
                    <div class="caption">Segmentation Results</div>
                </div>
            </div>
            <div class="image-container">
                <div class="image">
                    <img src="hemorrhage_3d_visualization.png" width="500">
                    <div class="caption">3D Visualization</div>
                </div>
            </div>
            <div class="image-container">
                <div class="image">
                    <img src="hemorrhage_heatmap.png" width="600">
                    <div class="caption">Hemorrhage Probability Heatmap</div>
                </div>
            </div>

            <h2>Analysis Notes</h2>
            <p>This analysis was performed using a multi-technique approach combining K-means clustering, 
               Random Forest classification, and Watershed segmentation. The segmentation was guided by 
               advanced image features including HOG, gradient magnitude, and texture descriptors.</p>

            <p>Generated by Advanced Brain Hemorrhage Analysis System</p>
        </body>
        </html>
        """

        # Save HTML report
        with open("hemorrhage_report.html", "w") as f:
            f.write(html_report)

        print(f"Report saved as 'hemorrhage_report.html'")
        return html_report


def main():
    """Main function to run the hemorrhage segmentation pipeline."""
    start_time = time.time()

    # File path
    image_path = 'butterfly.jpg'

    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found. Please check the path.")
        return

    try:
        # Initialize segmentation system
        segmenter = HemorrhageSegmentation(image_path)

        # Run preprocessing
        segmenter.preprocess()

        # Extract features
        segmenter.extract_features()

        # Perform K-means segmentation
        segmenter.segment_kmeans(n_clusters=4)

        # Perform watershed segmentation
        segmenter.segment_watershed()

        # Perform Random Forest segmentation
        segmenter.random_forest_segmentation()

        # Calculate metrics
        segmenter.calculate_metrics()

        # Visualize results
        segmenter.visualize_results()

        # Generate report
        segmenter.generate_report()

        elapsed_time = time.time() - start_time
        print(f"Total processing time: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()