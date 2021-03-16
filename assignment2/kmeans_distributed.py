import sys
import os
import logging
from typing import Dict
from mpi4py import MPI
import numpy as np
import pandas as pd

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
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size

        self.logger = ColorizedLog(logging.getLogger('KMeans Distributed %s' % self.rank),
                                   self.colors[self.rank])

    @staticmethod
    def _kmeans_log_setup():
        sys_path = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(sys_path, '..', 'logs', 'kmeans_internal_distributed.log')
        setup_log(log_path=log_path)

    @staticmethod
    def _chunk_list(seq, num):
        """Chunk a list into num parts.
        Args:
            seq: Any sequential type e.g. list, tuple etc
            num: Number of parts to chunk the list
        """

        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg
        return out

    @staticmethod
    def _chunk_for_scatterv(np_arr, size):
        avg_items_per_split, remaining_items = divmod(np_arr.shape[0], size)
        items_per_split = [avg_items_per_split + 1
                           if p < remaining_items else avg_items_per_split
                           for p in range(size)]
        items_per_split = np.array(items_per_split)
        # displacement: the starting index of each sub-task
        starting_index = [sum(items_per_split[:p]) for p in range(size)]
        starting_index = np.array(starting_index)
        return items_per_split, starting_index

    def _run_distributed(self, features: np.ndarray, num_clusters: int):
        """Run k-means algorithm to convergence.
    
        Args:
            features: numpy.ndarray: An N-by-d array describing N data points each of dimension d
            num_clusters: int: The number of clusters desired
        """

        # Scatter the points
        if self.rank == 0:
            self.logger.info(f"Started with {self.size} processes.")
            num_points = features.shape[0]  # num points
            num_features = features.shape[1]  # num features
            items_per_split_orig, starting_index_orig = self._chunk_for_scatterv(features, self.size)
            items_per_split = items_per_split_orig * num_features
            starting_index = starting_index_orig * num_features
            features_flat = features.flatten()  # Couldn't find a better way to scatter 2D np arrays
        else:
            num_points = None
            num_features = None
            features_flat = None
            # initialize items_per_split, and starting_index on worker processes
            items_per_split = np.zeros(self.size, dtype=np.int)
            items_per_split_orig = np.zeros(self.size, dtype=np.int)
            starting_index = None

        # Broadcast the number of items per split
        self.comm.Bcast(items_per_split, root=0)
        self.comm.Bcast(items_per_split_orig, root=0)
        num_points = self.comm.bcast(num_points, root=0)
        num_features = self.comm.bcast(num_features, root=0)

        # Scatter data points-features
        features_chunked_flat = np.zeros(items_per_split[self.rank])
        self.comm.Scatterv([features_flat, items_per_split, starting_index, MPI.DOUBLE],
                           features_chunked_flat,
                           root=0)
        features_chunked = features_chunked_flat.reshape(-1, num_features)

        # Initialize and Broadcast the Centroids
        if self.rank == 0:
            np.random.seed(0)
            centroid_ids = np.random.choice(num_points, size=(num_clusters,), replace=False)
            centroids = features[centroid_ids, :]

        else:
            centroids = np.zeros(num_points)
        centroids = self.comm.bcast(centroids, root=0)
        previous_cluster_assignments = np.zeros(num_points, dtype=np.uint8)

        # Loop until convergence
        while True:
            # Compute all-pairs distances from points to centroids
            centroid_distances_chunked = np.square(features_chunked[:, np.newaxis] - centroids) \
                .sum(axis=2)

            # Expectation step: assign clusters
            cluster_assignments_chunked = np.argmin(centroid_distances_chunked, axis=1)

            # Maximization step: Update centroid for each cluster
            for cluster_ind in range(num_clusters):
                features_of_curr_cluster = features_chunked[cluster_assignments_chunked == cluster_ind]
                # Find sum and count of each cluster
                count_curr_cluster_chunked = features_of_curr_cluster.shape[0]
                if count_curr_cluster_chunked > 0:
                    sum_curr_cluster_chunked = np.sum(features_of_curr_cluster, axis=0)
                else:
                    sum_curr_cluster_chunked = np.zeros_like(features_chunked[0])
                # Reduce the internal sums to find total sum
                sum_curr_cluster = np.zeros_like(sum_curr_cluster_chunked)
                # Find total sum for this cluster
                # self.logger.info(f"Chunked cluster sum: {sum_curr_cluster_chunked}")
                self.comm.Allreduce([sum_curr_cluster_chunked, MPI.DOUBLE],
                                 [sum_curr_cluster, MPI.DOUBLE],
                                 op=MPI.SUM)
                # Find total count for this cluster
                count_curr_cluster = self.comm.allreduce(count_curr_cluster_chunked, op=MPI.SUM)
                centroids[cluster_ind, :] = sum_curr_cluster/count_curr_cluster
            # Alternative: USE PANDAS TO GROUP BY CLUSTER -> MEAN ???

            # Break Condition
            # self.comm.Barrier()
            cluster_assignments = np.concatenate(self.comm.allgather(cluster_assignments_chunked))
            if (cluster_assignments == previous_cluster_assignments).all():
                break
            else:
                previous_cluster_assignments = cluster_assignments
        # return cluster centroids and cluster_assignments
        return centroids, cluster_assignments

    def run_distributed(self, num_clusters: int, dataset: str):
        if self.rank == 0:
            if dataset == 'iris':
                from sklearn.datasets import load_iris
                features, labels = load_iris(return_X_y=True)
            else:
                import pandas as pd

                # the directory contains a labels.csv which we will not need for clustering
                features_pd = pd.read_csv(dataset)
                features_pd.drop('Unnamed: 0', axis=1, inplace=True)
                # self.logger.info(f"Dataset columns: {features_pd.columns}")
                features = features_pd.to_numpy()
            self.logger.info(f"Dataset {dataset} loaded. Shape: {features.shape}.\n"
                             f"First Row: {features[0]}")
        else:
            features = None
        # run k-means
        # return
        centroids, assignments = self._run_distributed(features=features, num_clusters=num_clusters)

        # print out results
        if self.rank == 0:
            self.logger.info(f"\nCentroids: {centroids}\nAssignments: {assignments}")


if __name__ == '__main__':
    # take arguments like number of clusters k
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, required=True, help='Number of clusters')
    parser.add_argument('-d', type=str, required=False, default='iris', help='Dataset to use')
    args = parser.parse_args()
    num_clusters = int(args.k)
    dataset = args.d
    kmeans_runner = KMeansRunner()
    kmeans_runner.run_distributed(num_clusters=num_clusters, dataset=dataset)
