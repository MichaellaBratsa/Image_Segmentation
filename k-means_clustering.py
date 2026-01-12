import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')


def kmeans(data, K, thresh, n_iter, n_attempts):
    """
    Cluster data into K clusters using the K-Means algorithm.
    :param data: numpy.array(float), the input data array with N (#data) x D (#feature_dimensions) dimensions
    :param K: int, number of clusters
    :param thresh: float, convergence threshold
    :param n_iter: int, #iterations of the K-Means algorithm
    :param n_attempts: int, #attempts to run the K-Means algorithm
    :return:
    compactness: float, the sum of squared distance from each point to their corresponding centers
    labels: numpy.array(int), the label array with Nx1 dimensions, where it denotes the corresponding cluster of
    each data point
    centers : numpy.array(float), a KxD array with the final centroids
    """

    # Checks
    assert data.ndim == 2
    assert K > 0
    assert n_iter > 0
    assert n_attempts > 0
    assert n_attempts >= K

    best_comp = float('inf')
    best_labels = None
    best_centers = None
    labels = []

    for _ in range(n_attempts):

        # Choose random points as centroids
        indices = np.random.choice(data.shape[0], size=K, replace=False)
        centers = data[indices]
        new_centers = centers.copy()

        for _ in range(n_iter):

            # Compute distances from each point to each centroid
            euclidean_distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)

            # Assign labels to nearest center
            labels = np.argmin(euclidean_distances, axis=1)

            # Update centers
            for k in range(K):
                cluster_points = data[labels == k]
                if len(cluster_points) > 0:
                    new_centers[k] = cluster_points.mean(axis=0)
                else:
                    # Reassign empty centroid to a random data point
                    new_centers[k] = data[np.random.choice(data.shape[0])]

            # Calculate shift and check for convergence
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers.copy()

            if shift < thresh:
                break

        # Compute compactness
        compactness = np.sum((data - centers[labels]) ** 2)

        if compactness < best_comp:
            best_comp = compactness
            best_labels = labels
            best_centers = centers

    return best_comp, best_labels, best_centers


def main():
    # Load image
    image = cv2.imread('data/home.jpg')
    height, width = image.shape[:2]

    # Plot original image
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Home - Original Image")
    plt.axis('off')

    # Define parameters
    K = 4
    thresh = 1.0
    n_iter = 10
    n_attempts = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, n_iter, thresh)

    rgb_vectors = image.reshape(-1, 3).astype(np.float32)

    # Input features rgb pixel values

    plt.figure(figsize=(12, 5))

    # Personal implementation
    compactness, label, center = kmeans(rgb_vectors, K, thresh, n_iter, n_attempts)
    center = np.uint8(center)
    result = center[label.flatten()].reshape(image.shape)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Personal Implementation")
    plt.axis('off')

    # OpenCV implementation
    compactness_cv2, label_cv2, center_cv2 = cv2.kmeans(rgb_vectors, K, None, criteria, n_attempts, cv2.KMEANS_RANDOM_CENTERS)
    center_cv2 = np.uint8(center_cv2)
    result_cv2 = center_cv2[label_cv2.flatten()].reshape(image.shape)

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_cv2, cv2.COLOR_BGR2RGB))
    plt.title("OpenCV Implementation")
    plt.axis('off')

    # Input features pixels coordinates and values
    plt.figure(figsize=(12, 5))

    i_coords, j_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    i_coords = i_coords.reshape(-1, 1).astype(np.float32)
    j_coords = j_coords.reshape(-1, 1).astype(np.float32)

    # Combine spatial and color features
    features = np.hstack((i_coords, j_coords, rgb_vectors))

    # Personal implementation
    compactness, label, center = kmeans(features, K, thresh, n_iter, n_attempts)
    rgb_centers = center[:, 2:].astype(np.uint8)
    result = rgb_centers[label.flatten()].reshape(image.shape)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Personal Segmented Image (with Pixel Coordinates)")
    plt.axis('off')

    # OpenCV implementation
    compactness, label, center = cv2.kmeans(features, K, None, criteria, n_attempts, cv2.KMEANS_RANDOM_CENTERS)
    rgb_centers = center[:, 2:].astype(np.uint8)
    result = rgb_centers[label.flatten()].reshape((height, width, 3))

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Segmented Image (with Pixel Coordinates)")
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()
