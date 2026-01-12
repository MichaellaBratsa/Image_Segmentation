import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import matplotlib

matplotlib.use('TkAgg')


def nn_graph(input_image, k):
    assert len(input_image.shape) == 3 and input_image.shape[2] == 3, 'Input image must be 3 dimensional'
    assert k > 0, 'k must be > 0'

    H = input_image.shape[0]
    W = input_image.shape[1]

    coords = np.indices((H, W)).transpose(1, 2, 0).reshape(-1, 2)
    rgb = input_image.reshape(-1, 3).astype(np.float32)
    features = np.hstack((coords.astype(np.float32), rgb))

    # Initialize the k-nearest neighbors using Euclidean distance
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(features)

    # Find the k-nearest neighbors
    distances, indices = nn.kneighbors(features)

    edges = []
    for src_idx, (neighbors, dists) in enumerate(zip(indices, distances)):
        for dst_idx, dist in zip(neighbors[1:], dists[1:]):
            edges.append((src_idx, dst_idx, dist))

    return coords, edges


def find_root(sets, i):
    if sets[i] != i:
        sets[i] = find_root(sets, sets[i])
    return sets[i]


def union(sets, sizes, internal_diff, u, v, w, k):
    u_root = find_root(sets, u)
    v_root = find_root(sets, v)

    if u_root == v_root:
        return False

    t_u = k / sizes[u_root]
    t_v = k / sizes[v_root]

    # Minimum internal difference
    MInt = min(internal_diff[u_root] + t_u, internal_diff[v_root] + t_v)

    if w <= MInt:
        if sizes[u_root] < sizes[v_root]:
            u_root, v_root = v_root, u_root

        sets[v_root] = u_root
        sizes[u_root] += sizes[v_root]
        internal_diff[u_root] = max(internal_diff[u_root], internal_diff[v_root], w)
        return True

    return False


def segmentation(G, k, min_size):
    """
    Segment the image base on the Efficient Graph-Based Image Segmentation algorithm.
    :param G: tuple(V, E), the input graph
    :param k: int, sets the threshold k/|C|
    :param min_size: int, minimum size of clusters
    :return:
    clusters: numpy.array(int), a |V|x1 array where it denotes the cluster for each node v of the graph
    """

    assert k > 0 and min_size > 0, "k and min_size must be positive non-zero numbers"

    V = G[0]
    E = G[1]

    num_pixels = V.shape[0]

    # Sort edges in ascending order by edge weight
    sorted_edges = sorted(E, key=lambda x: x[2])

    # Initialize one set per pixel
    sets = np.arange(num_pixels)
    sizes = np.ones(num_pixels, dtype=np.int32)
    internal_diff = np.zeros(num_pixels, dtype=np.float32)


    for u, v, w in sorted_edges:
        union(sets, sizes, internal_diff, u, v, w, k)

    for u, v, w in sorted_edges:
        u_root = find_root(sets, u)
        v_root = find_root(sets, v)

        if u_root != v_root:
            if sizes[u_root] < min_size or sizes[v_root] < min_size:
                sets[v_root] = u_root
                sizes[u_root] += sizes[v_root]

    # Assign final cluster labels
    clusters = np.zeros(num_pixels, dtype=np.int32)
    root_to_cluster = {}
    cluster_id = 0
    for i in range(num_pixels):
        root = find_root(sets, i)
        if root not in root_to_cluster:
            root_to_cluster[root] = cluster_id
            cluster_id += 1
        clusters[i] = root_to_cluster[root]

    return clusters


def color_clusters(clusters, H, W):
    num_clusters = clusters.max() + 1

    hsv_colors = np.zeros((num_clusters, 3), dtype=np.float32)
    hsv_colors[:, 0] = np.linspace(0, 1, num_clusters, endpoint=False)  # Hue
    hsv_colors[:, 1] = 1.0  # Saturation
    hsv_colors[:, 2] = 1.0  # Value

    rgb_colors = cv2.cvtColor((hsv_colors.reshape(1, -1, 3) * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)[0]

    # Map each cluster into a color
    segmented_img = rgb_colors[clusters].reshape((H, W, 3))

    return segmented_img


def main():
    image = cv2.imread('data/eiffel_tower.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    H = image.shape[0]
    W = image.shape[1]

    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.figure()

    # Apply Gaussian blur
    smooth_img = cv2.GaussianBlur(image, (3, 3), 0.8)

    # Construct the graph
    V, E = nn_graph(smooth_img, k=10)

    clusters = segmentation((V, E), k=550, min_size=300)
    colored_segmentation = color_clusters(clusters, H, W)

    # Plot results
    plt.imshow(colored_segmentation)
    plt.title("Segmented Image")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
