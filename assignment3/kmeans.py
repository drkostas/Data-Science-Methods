import os
from typing import Dict, Callable
from mpi4py import MPI
import numpy as np

from playground import ColorizedLogger, timeit


class KMeansRunner:
    run_type: str
    logger: ColorizedLogger
    run_func: Callable
    features_iris: np.ndarray
    features_tcga: np.ndarray

    def __init__(self, run_type: str):
        funcs = {'simple': self._run_simple,
                 'vectorized': self._run_vectorized,
                 'vectorized_jacob': self._run_vectorized_jacob}
        self.run_type = run_type
        self.run_func = funcs[self.run_type]
        self.features_iris = None
        self.features_tcga = None
        self.logger = ColorizedLogger(f'KMeans {run_type}', 'green')

    @staticmethod
    def _run_simple(features: np.ndarray, num_clusters: int):
        """Run Simple K-Means algorithm to convergence.

        Args:
            features: numpy.ndarray: An N-by-d array describing N data points each of dimension d
            num_clusters: int: The number of clusters desired
        """

        N = features.shape[0]  # num sample points
        d = features.shape[1]  # dimension of space

        #
        # INITIALIZATION PHASE
        # initialize centroids randomly as distinct elements of xs
        np.random.seed(0)
        cids = np.random.choice(N, (num_clusters,), replace=False)
        centroids = features[cids, :]
        assignments = np.zeros(N, dtype=np.uint8)

        # loop until convergence
        while True:
            # Compute distances from sample points to centroids
            # all  pair-wise _squared_ distances
            cdists = np.zeros((N, num_clusters))
            for i in range(N):
                xi = features[i, :]
                for c in range(num_clusters):
                    cc = centroids[c, :]
                    dist = 0
                    for j in range(d):
                        dist += (xi[j] - cc[j]) ** 2
                    cdists[i, c] = dist

            # Expectation step: assign clusters
            num_changed_assignments = 0
            for i in range(N):
                # pick closest cluster
                cmin = 0
                mindist = np.inf
                for c in range(num_clusters):
                    if cdists[i, c] < mindist:
                        cmin = c
                        mindist = cdists[i, c]
                if assignments[i] != cmin:
                    num_changed_assignments += 1
                assignments[i] = cmin

            # Maximization step: Update centroid for each cluster
            for c in range(num_clusters):
                newcent = 0
                clustersize = 0
                for i in range(N):
                    if assignments[i] == c:
                        newcent = newcent + features[i, :]
                        clustersize += 1
                newcent = newcent / clustersize
                centroids[c, :] = newcent

            if num_changed_assignments == 0:
                break

        # return cluster centroids and assignments
        return centroids, assignments

    def _run_vectorized_jacob(self, features: np.ndarray, num_clusters: int):

        """Run k-means algorithm to convergence.

        Args:
            features: numpy.ndarray: An num_points-by-d array describing num_points data points each
            of dimension d
            num_clusters: int: The number of clusters desired
        """
        num_points = features.shape[0]  # num sample points

        #
        # INITIALIZATION PHASE
        # initialize centroids randomly as distinct elements of xs
        np.random.seed(0)
        cids = np.random.choice(num_points, (num_clusters,), replace=False)
        centroids = features[cids, :]
        assignments = np.zeros(num_points, dtype=np.uint8)

        # loop until convergence
        loop_cnt = 0
        while True:
            loop_cnt += 1
            # Compute distances from sample points to centroids
            # all  pair-wise _squared_ distances
            centroid_distances = np.zeros((num_points, num_clusters))
            for i in range(num_points):
                xi = features[i, :]
                for c in range(num_clusters):
                    cc = centroids[c, :]

                    dist = np.sum((xi - cc) ** 2)

                    centroid_distances[i, c] = dist

            # Expectation step: assign clusters
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
                if assignments[i] != cmin:
                    num_changed_assignments += 1
                assignments[i] = cmin

            # Maximization step: Update centroid for each cluster
            for c in range(num_clusters):
                newcent = 0
                clustersize = 0
                for i in range(num_points):
                    if assignments[i] == c:
                        newcent = newcent + features[i, :]
                        clustersize += 1
                newcent = newcent / clustersize
                centroids[c, :] = newcent

            if num_changed_assignments == 0:
                break

        # return cluster centroids and assignments
        return centroids, assignments

    def _run_vectorized(self, features: np.ndarray, num_clusters: int):
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

        from scipy.spatial.distance import cdist

        num_points = features.shape[0]  # num sample points
        np.random.seed(0)
        centroid_ids = np.random.choice(num_points, (num_clusters,), replace=False)
        centroids = features[centroid_ids, :]
        cluster_assignments = np.zeros(num_points, dtype=np.uint8)
        # Loop until convergence
        loop_cnt = 0
        while True:
            loop_cnt += 1
            # Compute distances from sample points to centroids
            # all  pair-wise _squared_ distances
            centroid_distances = np.square(cdist(features, centroids, 'euclidean'))

            # Expectation step: assign clusters
            previous_assignments = cluster_assignments
            cluster_assignments = np.argmin(centroid_distances, axis=1)

            # Maximization step: Update centroid for each cluster
            for cluster_ind in range(num_clusters):
                features_of_curr_cluster = features[cluster_assignments == cluster_ind]
                centroids[cluster_ind, :] = np.mean(features_of_curr_cluster, axis=0)
            # USE PANDAS TO GROUP BY CLUSTER -> MEAN ???
            # Break Condition
            if (cluster_assignments == previous_assignments).all():
                break

        # return cluster centroids and cluster_assignments
        return centroids, cluster_assignments

    def run(self, num_clusters: int, dataset: str):
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

        dataset_name = 'tcga' if dataset != 'iris' else dataset

        if dataset == 'iris':
            if not self.features_iris:
                from sklearn.datasets import load_iris
                self.features_iris, _ = load_iris(return_X_y=True)
                self.logger.info(f"Dataset {dataset_name} loaded. Shape: {self.features_iris.shape}.")
            self.run_func(features=self.features_iris, num_clusters=num_clusters)
        else:
            if not self.features_tcga:
                import pandas as pd
                features_pd = pd.read_csv(dataset)
                features_pd.drop('Unnamed: 0', axis=1, inplace=True)
                self.features_tcga = features_pd.to_numpy()
                self.logger.info(f"Dataset {dataset_name} loaded. Shape: {self.features_tcga.shape}.")
            self.run_func(features=self.features_tcga, num_clusters=num_clusters)

        # Run K-Means and save results
        # sys_path = os.path.dirname(os.path.realpath(__file__))
        # output_file_name = f'assignment3_{dataset_name}_{self.run_type}.txt'
        # output_file_path = os.path.join(sys_path, '..', 'outputs')
        # if not os.path.exists(output_file_path):
        #     os.makedirs(output_file_path)
        # output_file_path = os.path.join(output_file_path, output_file_name)
        # with open(output_file_path, 'w') as f:
        #     f.write(f'K-Means {self.run_type} version for the {dataset_name} dataset '
        #             f'with {num_clusters} clusters and {self.size} process(es).\n')
        #     centroids, assignments = self.run_func(features=features,
        #                                                num_clusters=num_clusters)
        #     self.logger.info(f"Final Cluster Assignments: \n{assignments}")
        #     # Save results
        #     f.write(f'Assignments:\n')
        #     f.write(f'{assignments.tolist()}\n')
        #     f.write(f'Centroids:\n')
        #     f.write(f'{centroids.tolist()}')
