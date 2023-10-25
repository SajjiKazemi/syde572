import torch
from torchvision.datasets import MNIST

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Cluster:
    def __init__(self, init_vector):
        """
        A class representation of a cluster, used for hierarchal clustering

        Parameters
        ----------
        init_vector : np.ndarray
            Initial vector belonging to the cluster, shape (d,)

        """
        # What defines a cluster (in this case)?
        #   - a set of vectors which belong to the cluster (self.vectors)
        #   - a cluster center defined as the mean vector (self.center)
        self.vectors = np.asarray(init_vector[None])  # index with None to get shape (1, d)
        self.center = init_vector

    def merge(self, cluster):
        """
        Merges the given cluster with this cluster (self)

        Parameters
        ----------
        cluster : Cluster
            The cluster to merge

        """
        # Add the vectors from `cluster` to self.vectors
        self.vectors = np.concatenate((self.vectors, cluster.vectors), axis=0)

        # Re-compute the cluster center
        self.center = self.vectors.mean(axis=0)


def closest_clusters(clusters):
    """
    Identifies the closest pair of clusters based on
    Euclidean distances between cluster centers

    Parameters
    ----------
    clusters : list[Cluster]
        List of Cluster instances

    Returns
    -------
    i, j : int
        The indices of the two closest clusters in the list

    """
    centers = np.array([cluster.center for cluster in clusters])
    diffs = centers[None, :] - centers[:, None, :]
    dists = np.linalg.norm(diffs, axis=-1)
    idx1, idx2 = np.triu_indices(dists.shape[0], k=1)
    min_idx = np.argmin(dists[idx1, idx2])
    i, j = idx1[min_idx], idx2[min_idx]
    return i, j


def hierarchal_cluster(vectors, num_clusters):
    """Performs hierarchal clustering of a set of vectors

    Parameters
    ----------
    vectors : np.ndarray
        Array of d-dimensional vectors with shape (N, d)
    num_clusters : int
        The desired number of clusters

    Returns
    -------
    clusters : list[Cluster]
        List of Cluster instances with length num_clusters

    """
    # Create a cluster for each vector
    clusters = [Cluster(vec) for vec in vectors]
    curr_clusters = len(clusters)

    while curr_clusters > num_clusters:
        # Find indices of closest clusters
        i, j = closest_clusters(clusters)

        # Merge closest clusters
        clusters[i].merge(clusters[j])

        # Remove merged cluster from list
        clusters.pop(j)

        # We now have one less cluster
        curr_clusters -= 1
    return clusters

def check_image(image, clusters):
    """
    Checks the image to see if it is a 3 or a 7

    Parameters
    ----------
    image : torch.Tensor
        Image to check

    Returns
    -------
    is_3 : bool
        True if the image is a 3, False if the image is a 7

    """
    # Flatten image
    image = image.reshape(-1, 28*28).numpy()

    # Perform PCA
    pca = PCA(n_components=2)
    image_pca = pca.fit_transform(image).astype(np.float32)

    # Check which cluster the image belongs to
    #  - if the image is closer to the cluster with 3s, it is a 3
    # - if the image is closer to the cluster with 7s, it is a 7
    # - if the image is equally close to both clusters, it is a 3

    # Compute distances to cluster centers
    centers = np.array([cluster.center for cluster in clusters])
    diffs = centers[None, :] - image_pca[0, None, :]
    dists = np.linalg.norm(diffs, axis=-1)

    # Return True if the image is closer to the cluster with 3s
    return dists[0, 0] < dists[0, 1]

if __name__ == '__main__':

    root = 'data'
    classes = (3, 7)
    seed = 28  # for consistency between runs
    np.random.seed(seed)

    # Load data and select the classes of interest
    dataset = MNIST(root=root, train=True, download=True)
    subset_idx = torch.isin(dataset.targets, torch.as_tensor(classes))
    train_images = dataset.data[subset_idx].numpy()
    train_labels = dataset.targets[subset_idx].numpy()
    idx3 = train_labels == 3
    idx7 = ~idx3

    # Flatten images
    train_images_flat = train_images.reshape(-1, 28*28)

    # Perform PCA with 2 components, grab 500 random points
    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(train_images_flat).astype(np.float32)

    # This implementation is not very efficient, so we'll use 500 random points
    np.random.shuffle(train_pca)
    train_pca = train_pca[:500]

    # Perform hierarchal clustering
    n_clusts = 2  # change as desired to see the behaviour
    clusters = hierarchal_cluster(train_pca, n_clusts)

    # Show clusters
    plt.figure()
    for i, cluster in enumerate(clusters):
        plt.plot(
            cluster.vectors[:, 0], cluster.vectors[:, 1], 'o',
            markerfacecolor='none', label='Cluster {}'.format(i + 1)
        )
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.show()

    plt.imshow(dataset.data[13].reshape(28, 28), cmap='gray')
    plt.imshow(dataset.data[11].reshape(28, 28), cmap='gray')
    if check_image(dataset.data[10:12], clusters):
        print('The image is a 3')
    else:
        print('The image is a 7')
    
    plt.imshow(dataset.data[15].reshape(28, 28), cmap='gray')
    plt.imshow(dataset.data[16].reshape(28, 28), cmap='gray')
    if check_image(dataset.data[15:17], clusters):
        print('The image is a 3')
    else:
        print('The image is a 7')
