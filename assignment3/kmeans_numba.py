import os
import numpy as np
from numba import jit, prange

from playground import profileit
from kmeans import KMeansRunner


@jit(nopython=True, parallel=True)
def _compute_distances_vectorized_jacob(num_points: int, num_clusters: int,
                                        centroids: np.ndarray, features: np.ndarray):
    # all  pair-wise _squared_ distances
    centroid_distances = np.zeros((num_points, num_clusters))
    for i in prange(num_points):
        xi = features[i, :]
        for c in prange(num_clusters):
            cc = centroids[c, :]
            dist = np.sum((xi - cc) ** 2)
            centroid_distances[i, c] = dist
    return centroid_distances


@jit(nopython=True, parallel=True)
def _expectation_step_vectorized_jacob(num_points: int, num_clusters: int,
                                       centroid_distances: np.ndarray,
                                       cluster_assignments: np.ndarray):
    num_changed_assignments = 0
    # claim: we can just do the following:
    # assignments = np.argmin(centroid_distances, axis=1)
    for i in prange(num_points):
        # pick closest cluster
        cmin = 0
        mindist = np.inf
        for c in prange(num_clusters):
            if centroid_distances[i, c] < mindist:
                cmin = c
                mindist = centroid_distances[i, c]
        if cluster_assignments[i] != cmin:
            num_changed_assignments += 1
        cluster_assignments[i] = cmin

    return centroid_distances, cluster_assignments, num_changed_assignments


@jit(nopython=True, parallel=True)
def _maximization_step_vectorized_jacob(num_clusters: int, num_points: int,
                                        cluster_assignments: np.ndarray,
                                        features: np.ndarray, centroids: np.ndarray):
    for c in prange(num_clusters):
        new_centroid = np.ones(1, )
        cluster_size = 0
        for i in prange(num_points):
            if cluster_assignments[i] == c:
                new_centroid = new_centroid + features[i, :]
                cluster_size += 1
        new_centroid = new_centroid / cluster_size
        centroids[c, :] = new_centroid

    return centroids


@jit(nopython=True)
def _loop_vectorized_jacob(num_clusters: int, num_points: int, cluster_assignments: np.ndarray,
                           features: np.ndarray, centroids: np.ndarray):
    loop_cnt = 0
    while True:
        loop_cnt += 1
        # Compute distances from sample points to centroids
        centroid_distances = _compute_distances_vectorized_jacob(num_points,
                                                                 num_clusters,
                                                                 centroids, features)

        # Expectation step: assign clusters
        centroid_distances, cluster_assignments, num_changed_assignments = \
            _expectation_step_vectorized_jacob(num_points, num_clusters,
                                               centroid_distances,
                                               cluster_assignments)

        # Maximization step: Update centroid for each cluster
        centroids = _maximization_step_vectorized_jacob(num_clusters, num_points,
                                                        cluster_assignments,
                                                        features, centroids)

        if num_changed_assignments == 0:
            break

    # return cluster centroids and assignments
    return centroids, cluster_assignments


@jit(nopython=True)
def run_vectorized_jacob(features: np.ndarray, num_clusters: int):
    """Run k-means algorithm to convergence.

    Args:
        features: numpy.ndarray: An num_points-by-d array describing num_points data points each
        of dimension d
        num_clusters: int: The number of clusters desired
    """
    num_points = features.shape[0]  # num sample points

    # INITIALIZATION PHASE
    # initialize centroids randomly as distinct elements of xs
    np.random.seed(0)
    centroids_ids = np.random.choice(num_points, (num_clusters,), replace=False)
    centroids = features[centroids_ids, :]
    cluster_assignments = np.zeros(num_points, dtype=np.uint8)

    # loop until convergence
    centroids, cluster_assignments = \
        _loop_vectorized_jacob(num_clusters, num_points, cluster_assignments, features, centroids)

    # return cluster centroids and assignments
    return centroids, cluster_assignments


def run(kmeans_obj: KMeansRunner, run_type: str, num_clusters: int, dataset: str):
    """

    Args:
        kmeans_obj:
        num_clusters: The number of clusters to find
        dataset: The name or path of the dataset

    Returns:

    Info:
        features shape: (# points, # features)
        centroids shape: (# clusters, # features)
        centroid_distances shape: (# points, # clusters)
    """

    # Setup func to run and dataset to use
    dataset_name = 'tcga' if dataset != 'iris' else dataset

    # Prepare output folders and names
    sys_path = os.path.dirname(os.path.realpath(__file__))
    output_file_name = f'assignment3_{dataset_name}_{run_type}.txt'
    profiler_file_name = f'assignment3_{dataset_name}_{run_type}.o'
    output_base_path = os.path.join(sys_path, '..', 'outputs')
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    profiler_file_path = os.path.join(output_base_path, profiler_file_name)
    output_file_path = os.path.join(output_base_path, output_file_name)

    # Open results output file
    with open(output_file_path, 'w') as outputs_file:
        outputs_file.write(f'K-Means {run_type} version for the {dataset_name} dataset '
                           f'with {num_clusters} clusters .\n')

        # Load Dataset if not already loaded
        features = kmeans_obj._load_dataset(dataset_name, dataset)

        # Run Kmeans
        k_words = ['kmeans_numba.py', 'ncalls']  # Include only pstats that contain these words
        custom_print = f'Profiling `{run_type}` K-Means for the `{dataset_name}` dataset: '
        with profileit(file=outputs_file, profiler_output=profiler_file_path,
                       custom_print=custom_print,
                       keep_only_these=k_words):
            centroids, assignments = run_vectorized_jacob(features=features, num_clusters=num_clusters)

        # Save results
        kmeans_obj.logger.info(f"Final Cluster Assignments: \n{assignments}")
        outputs_file.write(f'Assignments:\n')
        outputs_file.write(f'{assignments.tolist()}\n')
        outputs_file.write(f'Centroids:\n')
        outputs_file.write(f'{centroids.tolist()}')
