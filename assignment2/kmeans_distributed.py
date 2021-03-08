"""Vectorized k-means implementation for DSE512"""

import numpy as np
from mpi4py import MPI

# find MPI info
comm = MPI.COMM_WORLD
world_size =  comm.Get_size()
rank =  comm.Get_rank()

def kmeans_distributed(xs, initial_centroids, num_clusters=4):
    """Run k-means algorithm to convergence.

    Args:
        xs: numpy.ndarray: An N-by-d array describing N data points each of dimension d
        num_clusters: int: The number of clusters desired
    """
    N = xs.shape[0]  # num sample points
    d = xs.shape[1]  # dimension of space

    centroids = initial_centroids
    assignments = np.zeros(N, dtype=np.uint8)

    # loop until convergence
    while True:
        # Compute distances from sample points to centroids
        # all  pair-wise _squared_ distances
        cdists = np.zeros((N, num_clusters))
        for i in range(N):
            xi = xs[i, :]
            for c in range(num_clusters):
                cc  = centroids[c, :]

                dist = np.sum((xi - cc) ** 2)

                cdists[i, c] = dist

        # Expectation step: assign clusters
        num_changed_assignments = 0
        # claim: we can just do the following:
        #assignments = np.argmin(cdists, axis=1)
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
                    newcent = newcent + xs[i, :]
                    clustersize += 1

            clustersize = comm.allreduce(clustersize, op=MPI.SUM)
            newcent = comm.allreduce(newcent, op=MPI.SUM)

            # avoid divide by zero
            if clustersize == 0:
                clustersize = 1
            newcent = newcent / clustersize
            centroids[c, :]  = newcent

        # convergence check should apply across _all_ ranks, so we need to sum
        # all of the number of changed cluster assignments, across partitions
        num_changed_assignments = comm.allreduce(num_changed_assignments, op=MPI.SUM)

        if num_changed_assignments == 0:
            break

    # return cluster centroids and assignments
    return centroids, assignments


if __name__ == '__main__':
    # take arguments like number of clusters k
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, required=True, help='Number of clusters')
    args = parser.parse_args()

    # rank 0 is typically referred to as a "master" rank
    if rank == 0:
        # load some sample data
        from sklearn.datasets import load_iris
        features, labels = load_iris(return_X_y=True)
        #
        # INITIALIZATION PHASE
        # initialize centroids randomly as distinct elements of xs
        np.random.seed(0)
        cids = np.random.choice(features.shape[0], (args.k,), replace=False)
        initial_centroids  = features[cids, :]
        initial_centroids = comm.bcast(initial_centroids, root=0)

        # scatter features across ranks
        features = np.array_split(features, world_size, axis=0)
        myfeatures = comm.scatter(features, root=0)
    else:
        initial_centroids = comm.bcast(None, root=0)
        myfeatures = comm.scatter(None, root=0)

    # run k-means
    centroids, assignments = kmeans_distributed(myfeatures, initial_centroids, num_clusters=args.k)  

    # print out results only on one rank (reduces clutter in output)
    if rank == 0:
        print(f"rank={rank}",
            f"centroids={centroids}",
            f"assignments={assignments}")
