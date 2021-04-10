import os
import numpy as np
from numba import jit
from typing import IO

from playground import timeit
from kmeans import KMeansRunner


@jit(nopython=True)
def _compute_distances_vectorized_jacob(num_points: int, num_clusters: int,
                                        centroids: np.ndarray, features: np.ndarray):
    # all  pair-wise _squared_ distances
    centroid_distances = np.zeros((num_points, num_clusters))
    for i in range(num_points):
        xi = features[i, :]
        for c in range(num_clusters):
            cc = centroids[c, :]
            dist = np.sum((xi - cc) ** 2)
            centroid_distances[i, c] = dist
    return centroid_distances


@jit(nopython=True)
def _expectation_step_vectorized_jacob(num_points: int, num_clusters: int,
                                       centroid_distances: np.ndarray,
                                       cluster_assignments: np.ndarray):
    num_changed_assignments = 0
    # claim: we can just do the following:
    # assignments = np.argmin(centroid_distances, axis=1)
    for i in range(num_points):
        # pick closest cluster
        cmin = 0
        mindist = np.inf
        for c in range(num_clusters):
            if centroid_distances[i, c] < mindist:
                cmin = c
                mindist = centroid_distances[i, c]
        if cluster_assignments[i] != cmin:
            num_changed_assignments += 1
        cluster_assignments[i] = cmin

    return centroid_distances, cluster_assignments, num_changed_assignments


@jit(nopython=True)
def _maximization_step_vectorized_jacob(num_clusters: int, num_points: int,
                                        cluster_assignments: np.ndarray,
                                        features: np.ndarray, centroids: np.ndarray):
    for c in range(num_clusters):
        new_centroid = np.ones(1, )
        cluster_size = 0
        for i in range(num_points):
            if cluster_assignments[i] == c:
                new_centroid = new_centroid + features[i, :]
                cluster_size += 1
        new_centroid = new_centroid / cluster_size
        centroids[c, :] = new_centroid

    return centroids


def _loop_vectorized_jacob(num_clusters: int, num_points: int, cluster_assignments: np.ndarray,
                           features: np.ndarray, centroids: np.ndarray, outputs_file: IO):
    loop_cnt = 0
    t_time_dists = 0.0
    t_time_expect = 0.0
    t_time_maxim = 0.0
    while True:
        loop_cnt += 1
        # Compute distances from sample points to centroids
        timeit_obj = timeit(internal_only=True)
        with timeit_obj:
            centroid_distances = _compute_distances_vectorized_jacob(num_points,
                                                                     num_clusters,
                                                                     centroids, features)
        t_time_dists += timeit_obj.total

        # Expectation step: assign clusters
        timeit_obj = timeit(internal_only=True)
        with timeit_obj:
            centroid_distances, cluster_assignments, num_changed_assignments = \
                _expectation_step_vectorized_jacob(num_points, num_clusters,
                                                   centroid_distances,
                                                   cluster_assignments)
        t_time_expect += timeit_obj.total

        # Maximization step: Update centroid for each cluster
        timeit_obj = timeit(internal_only=True)
        with timeit_obj:
            centroids = _maximization_step_vectorized_jacob(num_clusters, num_points,
                                                            cluster_assignments,
                                                            features, centroids)
        t_time_maxim += timeit_obj.total

        if num_changed_assignments == 0:
            break

    custom_print = '_compute_distances_vectorized_jacob K-Means using numba' + \
                   f' dataset took {t_time_dists:.4f} sec(s)\n'
    outputs_file.write(custom_print)
    custom_print = '_expectation_step_vectorized_jacob K-Means using numba' + \
                   f' dataset took {t_time_expect:.4f} sec(s)\n'
    outputs_file.write(custom_print)
    custom_print = '_maximization_step_vectorized_jacob K-Means using numba' + \
                   f' dataset took {t_time_maxim:.4f} sec(s)\n'
    outputs_file.write(custom_print)

    # return cluster centroids and assignments
    return centroids, cluster_assignments


def run_vectorized_jacob(features: np.ndarray, num_clusters: int, outputs_file: IO):
    """Run k-means algorithm to convergence.

    Args:
        outputs_file:
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
    custom_print = f'_loop_vectorized_jacob K-Means using numba' + \
                   ' dataset took {duration:.4f} sec(s)\n'
    with timeit(file=outputs_file, custom_print=custom_print):
        centroids, cluster_assignments = \
            _loop_vectorized_jacob(num_clusters, num_points, cluster_assignments,
                                   features, centroids, outputs_file)

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
    output_base_path = os.path.join(sys_path, '..', 'outputs')
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    output_file_path = os.path.join(output_base_path, output_file_name)

    # Open results output file
    outputs_file = open(output_file_path, 'w')
    outputs_file.write(f'K-Means {run_type} version for the {dataset_name} dataset '
                       f'with {num_clusters} clusters .\n')

    # Load Dataset if not already loaded
    features = kmeans_obj._load_dataset(dataset_name, dataset)

    # Run Kmeans
    custom_print = f'`{run_type}` K-Means for the `{dataset_name}`' + \
                   ' dataset took {duration:.4f} sec(s)\n'
    with timeit(file=outputs_file, custom_print=custom_print):
        centroids, assignments = run_vectorized_jacob(features=features, num_clusters=num_clusters,
                                                      outputs_file=outputs_file)

    # Save results
    kmeans_obj.logger.info(f"Final Cluster Assignments: \n{assignments}")
    outputs_file.write(f'Assignments:\n')
    outputs_file.write(f'{assignments.tolist()}\n')
    outputs_file.write(f'Centroids:\n')
    outputs_file.write(f'{centroids.tolist()}')

    # Close file stream
    outputs_file.close()
