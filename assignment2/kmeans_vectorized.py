import sys
import os
import logging
from typing import Dict
from mpi4py import MPI
import numpy as np

from playground.fancy_log.colorized_log import ColorizedLog
from playground.main import setup_log


class KMeansRunner:
    __slots__ = ('comm', 'rank', 'size', 'logger', 'mpi_enabled')
    comm: MPI.COMM_WORLD
    rank: int
    size: int
    logger: ColorizedLog
    colors: Dict = {
        0: 'blue',
        1: 'green',
        2: 'magenta',
        3: 'cyan',
        4: 'yellow',
        5: 'white',
        6: 'grey',
        7: 'black'
    }

    def __init__(self):
        self._kmeans_log_setup()
        self.logger = ColorizedLog(logging.getLogger('Kmeans Vectorized'), self.colors[0])

    @staticmethod
    def _kmeans_log_setup():
        sys_path = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(sys_path, '..', 'logs', 'kmeans_internal_vectorized.log')
        setup_log(log_path=log_path)

    def _run_vectorized(self, features: np.ndarray, num_clusters: int):
        """Run k-means algorithm to convergence.

        This is the Lloyd's algorithm [2] which consists of alternating expectation
        and maximization steps.

        Args:
            features: numpy.ndarray: An num_features-by-d array describing num_features data points each of
                dimension d.
            num_clusters: int: The number of clusters desired.
        Returns:
            centroids: numpy.ndarray: A num_clusters-by-d array of cluster centroid
                positions.
            cluster_assignments: numpy.ndarray: An num_features-length vector of integers whose values
                from 0 to num_clusters-1 indicate which cluster each data element
                belongs to.

        [1] https://en.wikipedia.org/wiki/K-means_clustering
        [2] https://en.wikipedia.org/wiki/Lloyd%27s_algorithm
        """
        num_features = features.shape[0]  # num sample points
        #
        # INITIALIZATION PHASE
        # initialize centroids randomly as distinct elements of features
        np.random.seed(0)
        centroid_ids = np.random.choice(num_features, (num_clusters,), replace=False)
        centroids = features[centroid_ids, :]
        cluster_assignments = np.zeros(num_features, dtype=np.uint8)
        # Loop until convergence
        while True:
            # Compute distances from sample points to centroids
            # all  pair-wise _squared_ distances
            centroid_distances = np.square(features[:, np.newaxis] - centroids).sum(axis=2)

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

    def run_vectorized(self, num_clusters: int):
        from sklearn.datasets import load_iris
        features, labels = load_iris(return_X_y=True)

        # run k-means
        centroids, assignments = self._run_vectorized(features=features, num_clusters=num_clusters)

        # print out results
        self.logger.info(f"\nCentroids: {centroids}\nAssignments: {assignments}")


if __name__ == '__main__':
    # take arguments like number of clusters k
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, required=True, help='Number of clusters')
    args = parser.parse_args()
    num_clusters = int(args.k)
    kmeans_runner = KMeansRunner()
    kmeans_runner.run_vectorized(num_clusters=num_clusters)
