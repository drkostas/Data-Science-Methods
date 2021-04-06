import os
from typing import Dict, IO, Union, Callable
import numpy as np

from playground import ColorizedLogger, profileit


class KMeansRunner:
    logger: ColorizedLogger
    funcs: Dict
    outputs_file: IO
    features_iris: Union[np.ndarray, None]
    features_tcga: Union[np.ndarray, None]

    def __init__(self):
        self.funcs = {'simple': self._run_simple,
                      'vectorized': self._run_vectorized,
                      'vectorized_jacob': self._run_vectorized_jacob}
        self.features_iris = None
        self.features_tcga = None
        self.logger = ColorizedLogger(f'KMeans', 'green')

    @staticmethod
    def _compute_distances_simple(num_points: int, num_features: int, num_clusters: int,
                                  centroids: np.ndarray, features: np.ndarray):
        # all  pair-wise _squared_ distances
        centroid_distances = np.zeros((num_points, num_clusters))
        for i in range(num_points):
            xi = features[i, :]
            for c in range(num_clusters):
                cc = centroids[c, :]
                dist = 0
                for j in range(num_features):
                    dist += (xi[j] - cc[j]) ** 2
                centroid_distances[i, c] = dist

        return centroid_distances

    @staticmethod
    def _expectation_step_simple(num_points: int, num_clusters: int,
                                 centroid_distances: np.ndarray, cluster_assignments: np.ndarray):
        num_changed_assignments = 0
        for i in range(num_points):
            # pick closest cluster
            min_cluster = 0
            min_distance = np.inf
            for c in range(num_clusters):
                if centroid_distances[i, c] < min_distance:
                    min_cluster = c
                    min_distance = centroid_distances[i, c]
            if cluster_assignments[i] != min_cluster:
                num_changed_assignments += 1
            cluster_assignments[i] = min_cluster

        return cluster_assignments, num_changed_assignments

    @staticmethod
    def _maximization_step_simple(num_clusters: int, num_points: int, cluster_assignments: np.ndarray,
                                  features: np.ndarray, centroids: np.ndarray):
        for c in range(num_clusters):
            new_centroid = 0
            cluster_size = 0
            for i in range(num_points):
                if cluster_assignments[i] == c:
                    new_centroid = new_centroid + features[i, :]
                    cluster_size += 1
            new_centroid = new_centroid / cluster_size
            centroids[c, :] = new_centroid
        return centroids

    @staticmethod
    def _loop_simple(num_clusters: int, num_points: int, num_features: int,
                     cluster_assignments: np.ndarray, features: np.ndarray, centroids: np.ndarray):
        while True:
            # Compute distances from sample points to centroids
            centroid_distances = KMeansRunner._compute_distances_simple(num_points, num_features,
                                                                        num_clusters,
                                                                        centroids, features)

            # Expectation step: assign clusters
            cluster_assignments, \
            num_changed_assignments = KMeansRunner._expectation_step_simple(num_points,
                                                                            num_clusters,
                                                                            centroid_distances,
                                                                            cluster_assignments)

            # Maximization step: Update centroid for each cluster
            centroids = KMeansRunner._maximization_step_simple(num_clusters, num_points,
                                                               cluster_assignments, features,
                                                               centroids)

            if num_changed_assignments == 0:
                break

        # return cluster centroids and assignments
        return centroids, cluster_assignments

    @staticmethod
    def _run_simple(features: np.ndarray, num_clusters: int):
        """Run Simple K-Means algorithm to convergence.

        Args:
            features: numpy.ndarray: An N-by-d array describing N data points each of dimension d
            num_clusters: int: The number of clusters desired
        """

        num_points = features.shape[0]  # num sample points
        num_features = features.shape[1]  # num features

        # INITIALIZATION PHASE
        # initialize centroids randomly as distinct elements of xs
        np.random.seed(0)
        centroid_ids = np.random.choice(num_points, (num_clusters,), replace=False)
        centroids = features[centroid_ids, :]
        cluster_assignments = np.zeros(num_points, dtype=np.uint8)

        # loop until convergence
        centroids, cluster_assignments = \
            KMeansRunner._loop_simple(num_clusters, num_points, num_features, cluster_assignments,
                                      features, centroids)

        # return cluster centroids and assignments
        return centroids, cluster_assignments

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _maximization_step_vectorized_jacob(num_clusters: int, num_points: int,
                                            cluster_assignments: np.ndarray,
                                            features: np.ndarray, centroids: np.ndarray):
        for c in range(num_clusters):
            new_centroid = 0
            cluster_size = 0
            for i in range(num_points):
                if cluster_assignments[i] == c:
                    new_centroid = new_centroid + features[i, :]
                    cluster_size += 1
            new_centroid = new_centroid / cluster_size
            centroids[c, :] = new_centroid

        return centroids

    @staticmethod
    def _loop_vectorized_jacob(num_clusters: int, num_points: int, cluster_assignments: np.ndarray,
                               features: np.ndarray, centroids: np.ndarray):
        loop_cnt = 0
        while True:
            loop_cnt += 1
            # Compute distances from sample points to centroids
            centroid_distances = KMeansRunner._compute_distances_vectorized_jacob(num_points,
                                                                                  num_clusters,
                                                                                  centroids, features)

            # Expectation step: assign clusters
            centroid_distances, cluster_assignments, num_changed_assignments = \
                KMeansRunner._expectation_step_vectorized_jacob(num_points, num_clusters,
                                                                centroid_distances,
                                                                cluster_assignments)

            # Maximization step: Update centroid for each cluster
            centroids = KMeansRunner._maximization_step_vectorized_jacob(num_clusters, num_points,
                                                                         cluster_assignments,
                                                                         features, centroids)

            if num_changed_assignments == 0:
                break

        # return cluster centroids and assignments
        return centroids, cluster_assignments

    @staticmethod
    def _run_vectorized_jacob(features: np.ndarray, num_clusters: int):
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
            KMeansRunner._loop_vectorized_jacob(num_clusters, num_points, cluster_assignments,
                                                features, centroids)

        # return cluster centroids and assignments
        return centroids, cluster_assignments

    @staticmethod
    def _compute_distances_vectorized(centroids: np.ndarray, features: np.ndarray) -> np.ndarray:
        from scipy.spatial.distance import cdist
        # all  pair-wise _squared_ distances
        return np.square(cdist(features, centroids, 'euclidean'))

    @staticmethod
    def _expectation_step_vectorized(centroid_distances: np.ndarray,
                                     cluster_assignments: np.ndarray) -> [np.ndarray, np.ndarray]:
        return np.argmin(centroid_distances, axis=1), cluster_assignments

    @staticmethod
    def _maximization_step_vectorized(num_clusters: int, cluster_assignments: np.ndarray,
                                      features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        for cluster_ind in range(num_clusters):
            features_of_curr_cluster = features[cluster_assignments == cluster_ind]
            centroids[cluster_ind, :] = np.mean(features_of_curr_cluster, axis=0)
        # USE PANDAS TO GROUP BY CLUSTER -> MEAN ???
        return centroids

    @staticmethod
    def _loop_vectorized(num_clusters: int, cluster_assignments: np.ndarray,
                         features: np.ndarray, centroids: np.ndarray):
        loop_cnt = 0
        while True:
            loop_cnt += 1
            # Compute distances from sample points to centroids
            # all  pair-wise _squared_ distances
            centroid_distances = KMeansRunner._compute_distances_vectorized(centroids, features)

            # Expectation step: assign clusters
            cluster_assignments, previous_assignments = \
                KMeansRunner._expectation_step_vectorized(centroid_distances, cluster_assignments)

            # Maximization step: Update centroid for each cluster
            centroids = KMeansRunner._maximization_step_vectorized(num_clusters, cluster_assignments,
                                                                   features, centroids)
            # Break Condition
            if (cluster_assignments == previous_assignments).all():
                break

        # return cluster centroids and cluster_assignments
        return centroids, cluster_assignments

    @staticmethod
    def _run_vectorized(features: np.ndarray, num_clusters: int):
        """Run k-means algorithm to convergence.

            This is the Lloyd's algorithm [2] which consists of alternating expectation
            and maximization steps.

            Args:
                features: numpy.ndarray: An num_points-by-d array describing num_points data points
                each of dimension d.
                num_clusters: int: The number of clusters desired.
            Returns:
                centroids: numpy.ndarray: A num_clusters-by-d array of cluster centroid
                    positions.
                cluster_assignments: numpy.ndarray: An num_points-length vector of integers whose
                values from 0 to num_clusters-1 indicate which cluster each data element belongs to.

            [1] https://en.wikipedia.org/wiki/K-means_clustering
            [2] https://en.wikipedia.org/wiki/Lloyd%27s_algorithm
            """
        #
        # INITIALIZATION PHASE
        # initialize centroids randomly as distinct elements of features

        num_points = features.shape[0]  # num sample points
        np.random.seed(0)
        centroid_ids = np.random.choice(num_points, (num_clusters,), replace=False)
        centroids = features[centroid_ids, :]
        cluster_assignments = np.zeros(num_points, dtype=np.uint8)
        # Loop until convergence
        centroids, cluster_assignments = \
            KMeansRunner._loop_vectorized(num_clusters, cluster_assignments, features, centroids)

        # return cluster centroids and cluster_assignments
        return centroids, cluster_assignments

    def _load_dataset(self, dataset_name: str, dataset: str):
        if dataset == 'iris':
            if self.features_iris is None:
                from sklearn.datasets import load_iris
                self.features_iris, _ = load_iris(return_X_y=True)
                self.logger.info(
                    f"Dataset {dataset_name} loaded. Shape: {self.features_iris.shape}.")
            return self.features_iris
        else:
            if self.features_tcga is None:
                import pandas as pd
                features_pd = pd.read_csv(dataset)
                features_pd.drop('Unnamed: 0', axis=1, inplace=True)
                self.features_tcga = features_pd.to_numpy()
                self.logger.info(
                    f"Dataset {dataset_name} loaded. Shape: {self.features_tcga.shape}.")
            return self.features_tcga

    def run(self, run_type: str, num_clusters: int, dataset: str):
        """

        Args:
            num_clusters: The number of clusters to find
            dataset: The name or path of the dataset

        Returns:

        Info:
            features shape: (# points, # features)
            centroids shape: (# clusters, # features)
            centroid_distances shape: (# points, # clusters)
        """

        run_func = self.funcs[run_type]
        dataset_name = 'tcga' if dataset != 'iris' else dataset

        # Run K-Means and save results
        sys_path = os.path.dirname(os.path.realpath(__file__))
        output_file_name = f'assignment3_{dataset_name}_{run_type}.txt'
        profiler_file_name = f'assignment3_{dataset_name}_{run_type}.o'
        output_base_path = os.path.join(sys_path, '..', 'outputs')
        if not os.path.exists(output_base_path):
            os.makedirs(output_base_path)
        profiler_file_path = os.path.join(output_base_path, profiler_file_name)
        output_file_path = os.path.join(output_base_path, output_file_name)
        with open(output_file_path, 'w') as self.outputs_file:
            self.outputs_file.write(f'K-Means {run_type} version for the {dataset_name} dataset '
                                    f'with {num_clusters} clusters .\n')

            # Load Dataset
            features = self._load_dataset(dataset_name, dataset)

            # Run Kmeans
            k_words = ['kmeans.py', 'ncalls']
            custom_print = f'Profiling `{run_type}` K-Means for the `{dataset_name}` dataset: '
            with profileit(file=self.outputs_file, profiler_output=profiler_file_path,
                           custom_print=custom_print,
                           keep_only_these=k_words):
                centroids, assignments = run_func(features=features, num_clusters=num_clusters)

            # Save results
            self.logger.info(f"Final Cluster Assignments: \n{assignments}")
            self.outputs_file.write(f'Assignments:\n')
            self.outputs_file.write(f'{assignments.tolist()}\n')
            self.outputs_file.write(f'Centroids:\n')
            self.outputs_file.write(f'{centroids.tolist()}')
