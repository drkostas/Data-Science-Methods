import sys
import os
import logging
from typing import Dict
from mpi4py import MPI
from sklearn.datasets import load_iris
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

    def __init__(self, mpi):
        self._kmeans_log_setup()
        self.mpi_enabled = mpi
        if self.mpi_enabled:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.rank
            self.size = self.comm.size
            self.logger = ColorizedLog(logging.getLogger('Kmeans %s' % self.rank),
                                       self.colors[self.rank])
        else:
            self.logger = ColorizedLog(logging.getLogger('Kmeans Serial'), self.colors[0])

    @staticmethod
    def _kmeans_log_setup():
        sys_path = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(sys_path, '..', '..', 'logs', 'kmeans.log')
        setup_log(log_path=log_path)

    @staticmethod
    def _chunk_list(seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg
        return out

    """Simple k-means implementation for DSE512"""

    def _run_serial(self, features: np.ndarray, num_clusters: int):
        """Run k-means algorithm to convergence.

        Args:
            features: numpy.ndarray: An N-by-d array describing N data points each of dimension d
            num_clusters: int: The number of clusters desired
        """

        N = features.shape[0]  # num sample points
        d = features.shape[1]  # dimension of space

        #
        # INITIALIZATION PHASE
        # initialize centroids randomly as distinct elements of xs
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

    def run(self, num_clusters: int):
        from sklearn.datasets import load_iris
        features, labels = load_iris(return_X_y=True)

        # run k-means
        centroids, assignments = self._run_serial(features=features, num_clusters=num_clusters)

        # print out results
        self.logger.info(f"\nCentroids: {centroids}\nAssignments: {assignments}")


if __name__ == '__main__':
    num_clusters = int(sys.argv[1])
    if sys.argv[2] == 'serial':
        kmeans_runner = KMeansRunner(mpi=False)
    elif sys.argv[2] == 'mpi':
        kmeans_runner = KMeansRunner(mpi=True)
    else:
        raise Exception('Argument not recognized!')
    kmeans_runner.run(num_clusters=num_clusters)
