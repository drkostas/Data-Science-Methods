import os
import sys
from pprint import pprint
import logging
from typing import Dict, Callable
from mpi4py import MPI
import numpy as np

from playground.fancy_log.colorized_log import ColorizedLog
from playground.main import setup_log, timeit


class KMeansRunner:
    run_type: str
    rank: int
    size: int
    logger: ColorizedLog
    colors: Dict
    run_func: Callable

    def __init__(self, run_type: str):
        funcs = {'simple': self._run_simple,
                 'vectorized': self._run_vectorized,
                 'vectorized_jacob': self._run_vectorized_jacob,
                 'distributed': self._run_distributed,
                 'distributed_jacob': self._run_distributed_jacob}
        self.run_type = run_type
        self.run_func = funcs[self.run_type]
        if 'distributed' in self.run_type:
            self.comm: MPI.COMM_WORLD = MPI.COMM_WORLD
            self.rank: int = self.comm.rank
            self.size: int = self.comm.size
        else:
            self.rank = 0
            self.size = 1

        # Setup logger
        self.colors = {
            0: 'blue',
            1: 'green',
            2: 'magenta',
            3: 'cyan',
            4: 'yellow',
            5: 'white',
            6: 'grey',
            7: 'black'
        }
        if self.size > 8:
            for col_ind in range(8, self.size + 1):
                self.colors[col_ind] = 'green'
        self._kmeans_log_setup()
        self.logger = ColorizedLog(logging.getLogger(f'KMeans {run_type} Proc({self.rank})'),
                                   self.colors[self.rank])
        if self.rank == 0:
            self.logger.info(f"Started with {self.size} processes.")

    @staticmethod
    def _kmeans_log_setup():
        sys_path = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(sys_path, '..', 'logs', 'kmeans_internal_distributed.log')
        setup_log(log_path=log_path, mode='w')

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
            features: numpy.ndarray: An N-by-d array describing N data points each of dimension d
            num_clusters: int: The number of clusters desired
        """
        N = features.shape[0]  # num sample points

        #
        # INITIALIZATION PHASE
        # initialize centroids randomly as distinct elements of xs
        with timeit(custom_print='Init Time: {duration:2.5f} sec(s)', skip=self.rank != 0):
            np.random.seed(0)
            cids = np.random.choice(N, (num_clusters,), replace=False)
            centroids = features[cids, :]
            assignments = np.zeros(N, dtype=np.uint8)

        # loop until convergence
        loop_cnt = 0
        while True:
            loop_cnt += 1
            # Compute distances from sample points to centroids
            # all  pair-wise _squared_ distances
            with timeit(custom_print='First Loop Distances Calc Time: {duration:2.5f} sec(s)',
                        skip=(self.rank != 0 or loop_cnt != 1)):
                cdists = np.zeros((N, num_clusters))
                for i in range(N):
                    xi = features[i, :]
                    for c in range(num_clusters):
                        cc = centroids[c, :]

                        dist = np.sum((xi - cc) ** 2)

                        cdists[i, c] = dist

            with timeit(custom_print='First Loop Cluster Assignment Time: {duration:2.5f} sec(s)',
                        skip=(self.rank != 0 or loop_cnt != 1)):
                # Expectation step: assign clusters
                num_changed_assignments = 0
                # claim: we can just do the following:
                # assignments = np.argmin(cdists, axis=1)
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
            with timeit(custom_print='First Loop Centroids Update Time: {duration:2.5f} sec(s)',
                        skip=(self.rank != 0 or loop_cnt != 1)):
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

    def _run_distributed_jacob(self, features: np.ndarray, num_clusters: int):
        """Run k-means algorithm to convergence.

            Args:
                features: numpy.ndarray: An N-by-d array describing N data points each of dimension d
                num_clusters: int: The number of clusters desired
            """

        # rank 0 is typically referred to as a "master" rank
        if self.rank == 0:
            #
            # INITIALIZATION PHASE
            # initialize centroids randomly as distinct elements of xs
            np.random.seed(0)
            cids = np.random.choice(features.shape[0], (args.k,), replace=False)
            initial_centroids = features[cids, :]
            initial_centroids = self.comm.bcast(initial_centroids, root=0)

            # scatter features across ranks
            features = np.array_split(features, self.comm.Get_size(), axis=0)
            features = self.comm.scatter(features, root=0)
        else:
            initial_centroids = self.comm.bcast(None, root=0)
            features = self.comm.scatter(None, root=0)

        N = features.shape[0]  # num sample points

        centroids = initial_centroids
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

                    dist = np.sum((xi - cc) ** 2)

                    cdists[i, c] = dist

            # Expectation step: assign clusters
            num_changed_assignments = 0
            # claim: we can just do the following:
            # assignments = np.argmin(cdists, axis=1)
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

                clustersize = self.comm.allreduce(clustersize, op=MPI.SUM)
                newcent = self.comm.allreduce(newcent, op=MPI.SUM)

                # avoid divide by zero
                if clustersize == 0:
                    clustersize = 1
                newcent = newcent / clustersize
                centroids[c, :] = newcent

            # convergence check should apply across _all_ ranks, so we need to sum
            # all of the number of changed cluster assignments, across partitions
            num_changed_assignments = self.comm.allreduce(num_changed_assignments, op=MPI.SUM)

            if num_changed_assignments == 0:
                break

        # return cluster centroids and assignments
        return centroids, assignments

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
        #
        # INITIALIZATION PHASE
        # initialize centroids randomly as distinct elements of features
        with timeit(custom_print='Init Time: {duration:2.5f} sec(s)', skip=self.rank != 0):
            num_features = features.shape[0]  # num sample points
            np.random.seed(0)
            centroid_ids = np.random.choice(num_features, (num_clusters,), replace=False)
            centroids = features[centroid_ids, :]
            cluster_assignments = np.zeros(num_features, dtype=np.uint8)
        # Loop until convergence
        loop_cnt = 0
        while True:
            loop_cnt += 1
            # Compute distances from sample points to centroids
            # all  pair-wise _squared_ distances
            with timeit(custom_print='First Loop Distances Calc Time: {duration:2.5f} sec(s)',
                        skip=(self.rank != 0 or loop_cnt != 1)):
                centroid_distances = np.square(features[:, np.newaxis] - centroids).sum(axis=2)

            # Expectation step: assign clusters
            with timeit(custom_print='First Loop Cluster Assignment Time: {duration:2.5f} sec(s)',
                        skip=(self.rank != 0 or loop_cnt != 1)):
                previous_assignments = cluster_assignments
                cluster_assignments = np.argmin(centroid_distances, axis=1)

            # Maximization step: Update centroid for each cluster
            with timeit(custom_print='First Loop Centroids Update Time: {duration:2.5f} sec(s)',
                        skip=(self.rank != 0 or loop_cnt != 1)):
                for cluster_ind in range(num_clusters):
                    features_of_curr_cluster = features[cluster_assignments == cluster_ind]
                    centroids[cluster_ind, :] = np.mean(features_of_curr_cluster, axis=0)
            # USE PANDAS TO GROUP BY CLUSTER -> MEAN ???
            # Break Condition
            if (cluster_assignments == previous_assignments).all():
                break

        # return cluster centroids and cluster_assignments
        return centroids, cluster_assignments

    def _run_distributed(self, features: np.ndarray, num_clusters: int):
        """Run k-means algorithm to convergence.
    
        Args:
            features: numpy.ndarray: An N-by-d array describing N data points each of dimension d
            num_clusters: int: The number of clusters desired
        """

        with timeit(custom_print='Init Time: {duration:2.5f} sec(s)', skip=self.rank != 0):
            # Scatter the points
            if self.rank == 0:
                num_points = features.shape[0]  # num points
                num_features = features.shape[1]  # num features
                items_per_split_orig, starting_index_orig = self._chunk_for_scatterv(features,
                                                                                     self.size)
                items_per_split = items_per_split_orig * num_features
                starting_index = starting_index_orig * num_features
                features_flat = features.flatten()  # Couldn't find better way to scatter 2D np arrays
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
        loop_cnt = 0
        while True:
            loop_cnt += 1
            # Compute all-pairs distances from points to centroids
            with timeit(custom_print='First Loop Distances Calc Time: {duration:2.5f} sec(s)',
                        skip=(self.rank != 0 or loop_cnt != 1)):
                centroid_distances_chunked = np.square(features_chunked[:, np.newaxis] - centroids) \
                    .sum(axis=2)

            # Expectation step: assign clusters
            with timeit(custom_print='First Loop Cluster Assignment Time: {duration:2.5f} sec(s)',
                        skip=(self.rank != 0 or loop_cnt != 1)):
                cluster_assignments_chunked = np.argmin(centroid_distances_chunked, axis=1)

            # Maximization step: Update centroid for each cluster
            with timeit(custom_print='First Loop Centroids Update Time: {duration:2.5f} sec(s)',
                        skip=(self.rank != 0 or loop_cnt != 1)):
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
                    centroids[cluster_ind, :] = sum_curr_cluster / count_curr_cluster
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

    def run(self, num_clusters: int, dataset: str):
        dataset_name = 'tcga' if dataset != 'iris' else dataset
        sys_path = os.path.dirname(os.path.realpath(__file__))
        output_file_name = f'{dataset_name}_{self.run_type}_clust{num_clusters}.txt'
        output_file_path = os.path.join(sys_path, '..', 'outputs', output_file_name)
        if self.rank == 0:
            if dataset == 'iris':
                from sklearn.datasets import load_iris
                features, labels = load_iris(return_X_y=True)
            else:
                import pandas as pd
                features_pd = pd.read_csv(dataset)
                features_pd.drop('Unnamed: 0', axis=1, inplace=True)
                features = features_pd.to_numpy()
            self.logger.info(f"Dataset {dataset_name} loaded. Shape: {features.shape}.")
        else:
            features = None

        # Run K-Means and save results
        custom_print = 'Total Time: {duration:2.5f} sec(s)\n'
        if self.rank == 0:
            with open(output_file_path, 'w') as f:
                f.write(f'K-Means {self.run_type} version for the {dataset_name} dataset '
                        f'with {num_clusters} clusters and {self.size} processe(s).\n')
                # Run for rank=0
                with timeit(custom_print=custom_print, file=f):
                    centroids, assignments = self.run_func(features=features,
                                                           num_clusters=num_clusters)
                    self.logger.info(f"Final Cluster Assignments: \n{assignments}")
                # Save results
                f.write(f'Assignments:\n')
                f.write(f'{assignments.tolist()}\n')
                f.write(f'Centroids:\n')
                f.write(f'{centroids.tolist()}')
        else:
            # Run for other ranks
            self.run_func(features=features, num_clusters=num_clusters)


if __name__ == '__main__':
    # Read and Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, required=True, help='Number of clusters')
    parser.add_argument('-d', type=str, required=False, default='iris', help='Dataset to use')
    parser.add_argument('-t', type=str, required=True, help='Type of kmeans to run',
                        choices=['simple', 'vectorized', 'vectorized_jacob', 'distributed',
                                 'distributed_jacob'])
    args = parser.parse_args()
    kmeans_num_clusters = int(args.k)
    kmeans_dataset = args.d
    kmeans_type = args.t
    # Initialize and run K-Means
    kmeans_runner = KMeansRunner(run_type=kmeans_type)
    kmeans_runner.run(num_clusters=kmeans_num_clusters, dataset=kmeans_dataset)
