# Image Segmentation (K-Means & Graph-Based)

This project implements two classic image segmentation approaches: K-Means clustering and a simplified Efficient Graph-Based Image Segmentation method.

## 🚀 Features
### 1) K-Means Segmentation
- Custom K-Means implementation with multiple attempts and convergence thresholding
- Segmentation using different feature representations:
  - RGB pixel values (r,g,b)
  - Spatial + color features (i,j,r,g,b)
- Comparison with OpenCV’s kmeans implementation

### 2) Efficient Graph-Based Segmentation (Simplified)
- Gaussian smoothing pre-processing
- k-Nearest Neighbor graph construction in (i,j,r,g,b) feature space
- Edge weights based on Euclidean distance
- Graph-based merging using internal difference criteria
- Post-processing merge for small adjacent segments
- Cluster visualization with distinct colors (HSV mapping)

## 🛠️ Tech Stack
- Python
- NumPy
- OpenCV (validation / utilities)
- Matplotlib
- scikit-learn (NearestNeighbors)

## 📍 What This Project Demonstrates
- Unsupervised clustering for image segmentation
- Feature engineering for segmentation quality
- Graph construction in high-dimensional feature space
- Implementation of graph-based segmentation logic
- Visual evaluation and comparisons

## 📸 Results / Visualizations
- K-Means segmentation results (your implementation vs OpenCV)
- Graph-based segmentation output with colored clusters

<img width="1276" height="476" alt="Image_Segmentation" src="https://github.com/user-attachments/assets/db6002b6-2539-4257-be46-8910d68e348e" />


## 📄 Notes
This project was developed as part of a CS447: Computer Vision - University of Cyprus (Spring 2025) assignment focusing on segmentation techniques and algorithmic implementation.
