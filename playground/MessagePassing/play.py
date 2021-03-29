import sys
import os
from typing import Dict
from mpi4py import MPI
import numpy as np

from playground import ColorizedLogger


class MPlayI:
    __slots__ = ('comm', 'rank', 'size', 'logger')
    comm: MPI.COMM_WORLD
    rank: int
    size: int
    logger: ColorizedLogger
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
        self._mpi_log_setup()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size

        if self.rank == 0:
            self.logger.info(f"Starting with size: {self.size}")

    @staticmethod
    def _mpi_log_setup():
        sys_path = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(sys_path, '..', '..', 'logs', 'mpi.log')
        ColorizedLogger.setup_logger(log_path=log_path)

    @staticmethod
    def _chunk_list(seq, num):
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

    def simple(self):
        self.logger.info(f"Hello from rank {self.rank} of size {self.size}")
        # Wait for everyone to sync up
        self.comm.Barrier()

    def broadcast(self):

        if self.rank == 0:
            x = np.random.randn(4) * 100
        else:
            x = np.empty(4, dtype=np.float64)
            self.logger.info(x)

        self.logger.info(f"Rank {self.rank} before broadcast has {x}")
        self.comm.Bcast([x, MPI.DOUBLE])
        self.logger.info(f"Rank {self.rank} after broadcast has {x}")

    def scatter_gather(self):
        if self.rank == 0:
            data = [x for x in range(self.size)]
        else:
            data = None
        data = self.comm.scatter(data, root=0)
        self.logger.info(f"Rank {self.rank} after scatter has {data}")
        self.comm.Barrier()
        if self.rank == 0:
            self.logger.info("****Gathering!****")
        data = self.comm.gather(data, root=0)
        self.logger.info(f"Rank {self.rank} after gather has {data}")

    def all_gather(self):
        if self.rank == 0:
            data = [x for x in range(self.size)]
            self.logger.info(f"Rank {self.rank} scattering {data}")
        else:
            data = None

        data = self.comm.scatter(data, root=0)
        self.logger.info(f"Rank {self.rank} after scatter has {data}")

        self.comm.Barrier()

        if self.rank == 0:
            self.logger.info("****Gathering!****")

        # Note that we no longer specify the root here!
        data = self.comm.allgather(data)
        self.logger.info(f"Rank {self.rank} after gather has {data}")

    def mpi_reduce(self):
        if self.rank == 0:
            data = [x for x in range(1, self.size + 1)]
            self.logger.info(f"Rank {self.rank} scattering {data}")
        else:
            data = None

        data = self.comm.scatter(data, root=0)
        self.logger.info(f"Rank {self.rank} after scatter has {data}")

        self.comm.Barrier()

        if self.rank == 0:
            self.logger.info(f"****Reduce!****")

        data = self.comm.reduce(data, root=0)
        self.logger.info(f"Rank {self.rank} after reduce has {data}")

    def mpi_all_reduce(self):
        if self.rank == 0:
            data = [x for x in range(1, self.size + 1)]
            self.logger.info(f"Rank {self.rank} scattering {data}")
        else:
            data = None

        data = self.comm.scatter(data, root=0)
        self.logger.info(f"Rank {self.rank} after scatter has {data}")

        self.comm.Barrier()

        if self.rank == 0:
            self.logger.info(f"****Reduce!****")

        # Similar to allgather, we do no specify a root process!
        data = self.comm.allreduce(data)
        self.logger.info(f"Rank {self.rank} after reduce has {data}")

    def count_lines(self):
        """Count the total lines of files in specified folder"""
        if self.rank == 0:
            from glob import glob
            files_path = os.path.join('data', 'mpi_count_lines', '*.txt')
            files = list(glob(files_path))
            files = self._chunk_list(files, self.size)
        else:
            files = None
        files = self.comm.scatter(files, root=0)
        self.logger.info(f"Rank {self.rank} has to count lines for these files: {files}")
        lines_cnt = 0
        for file in files:
            with open(file, 'r') as f:
                lines_cnt += sum(1 for _ in f)
        self.logger.info(f"Rank {self.rank} counted {lines_cnt} lines in total")
        self.comm.Barrier()

        total_lines_cnt = self.comm.reduce(lines_cnt, root=0)
        if self.rank == 0:
            self.logger.info(f"After reduce, counted {total_lines_cnt} lines from all ranks.")

    def reduce_complex_old(self):

        if self.rank == 0:
            data = np.array([[x * 1.1, x * 1.1 + 2, x * 1.1 + 4, x * 1.1 + 6] for x in
                             range(1, (self.size + 1) * 2)])
            # Create output array of same size
            res = np.zeros_like(data[0])
            # Split input array by the number of available cores
            data_ch = np.array_split(data, self.size, axis=0)

            chunk_sizes = []

            for i in range(0, len(data_ch), 1):
                chunk_sizes = np.append(chunk_sizes, len(data_ch[i]))

            chunk_sizes_input = chunk_sizes * data.shape[1]
            displacements_input = np.insert(np.cumsum(chunk_sizes_input), 0, 0)[0:-1]

            chunk_sizes_output = chunk_sizes * data.shape[1]
            displacements_output = np.insert(np.cumsum(chunk_sizes_output), 0, 0)[0:-1]
            self.logger.info(f"Rank {self.rank} scattering {data_ch}")
            self.logger.info(f"Expected result format: {res}")
        else:
            # Create variables on other cores
            chunk_sizes_input = None
            displacements_input = None
            chunk_sizes_output = None
            displacements_output = None
            data_ch = None
            data = None
            res = None

        data_ch = self.comm.bcast(data_ch, root=0)  # Broadcast split array to other cores
        chunk_sizes_output = self.comm.bcast(chunk_sizes_output, root=0)
        displacements_output = self.comm.bcast(displacements_output, root=0)

        # Create array to receive subset of data on each core, where rank specifies the core
        output_chunk = np.zeros(np.shape(data_ch[self.rank]))
        self.comm.Scatterv([data, chunk_sizes_input, displacements_input, MPI.DOUBLE], output_chunk,
                           root=0)
        self.logger.info(
            f"Rank {self.rank} after scatter has (shape: {output_chunk.shape}):\n{output_chunk}")
        # Create output array on each core
        output = np.zeros([len(output_chunk), output_chunk.shape[1]])

        for i in range(0, np.shape(output_chunk)[0], 1):
            output[i, 0:output_chunk.shape[1]] = output_chunk[i]

        self.comm.Barrier()

        if self.rank == 0:
            self.logger.info(f"****Reduce!****")

        # self.comm.Gatherv(output, [res, chunk_sizes_output, displacements_output, MPI.DOUBLE],
        #                   root=0)  # Gather output data together
        output = np.mean(output, axis=0)
        self.logger.info(f"Rank {self.rank} output (shape: {output.shape}):\n{output}")
        self.comm.Reduce(
            output,
            [res, chunk_sizes_output, displacements_output, MPI.DOUBLE],
            op=MPI.SUM,
            root=0
        )

        self.logger.info(f"Rank {self.rank} after reduce has {res}")

    def reduce_complex(self):
        if self.rank == 0:
            data = np.array([[x * 1., (x + 2) * 1., (x + 4) * 1., (x + 6) * 1.] for x in
                             range(1, (self.size + 1) * 2)])
            cluster_assignments = np.array([0, 1, 1, 3, 1, 1, 2, 4, 1, 3, 3], dtype=np.int64)
            num_features = data.shape[1]
            items_per_split_orig, starting_index_orig = self._chunk_for_scatterv(data, self.size)
            items_per_split = items_per_split_orig * num_features
            starting_index = starting_index_orig * num_features
            self.logger.info(f"Data ({data.shape}): {data[:1]}, ..")
            data = data.flatten()
            self.logger.info(f"Data Flat "
                             f"({data.shape}, {data.dtype}):{data[:6]}, ..")
            self.logger.info(f"Assignments({cluster_assignments.shape}, {cluster_assignments.dtype}): "
                             f"{cluster_assignments}")
            self.logger.info(f"Items per split Original: {items_per_split_orig}")
            self.logger.info(f"Items per split: {items_per_split}")
            self.logger.info(f"Starting Index Original: {starting_index_orig}")
            self.logger.info(f"Starting Index: {starting_index}")
            self.logger.info(f"Num Features: {num_features}")
        else:
            data = None
            cluster_assignments = None
            num_features = None
            # initialize items_per_split, and starting_index on worker processes
            items_per_split = np.zeros(self.size, dtype=np.int)
            items_per_split_orig = np.zeros(self.size, dtype=np.int)
            starting_index = None
            starting_index_orig = None

        # Broadcast the number of items per split
        self.comm.Bcast(items_per_split, root=0)
        self.comm.Bcast(items_per_split_orig, root=0)
        num_features = self.comm.bcast(num_features, root=0)

        # Scatter cluster assignments
        cluster_assignments_chunked = np.zeros(items_per_split_orig[self.rank], dtype=np.int64)
        self.logger.info(f"Initialized chunked assignments ({cluster_assignments_chunked.shape}): "
                         f"{cluster_assignments_chunked}")
        self.comm.Scatterv([cluster_assignments, items_per_split_orig, starting_index_orig,
                            MPI.INT64_T], cluster_assignments_chunked, root=0)
        self.logger.info(f"Received cluster_assignments_chunked "
                         f"({cluster_assignments_chunked.shape}): {cluster_assignments_chunked}")

        # Scatter data points-features
        data_chunked_flat = np.zeros(items_per_split[self.rank])
        self.comm.Scatterv([data, items_per_split, starting_index, MPI.DOUBLE],
                           data_chunked_flat,
                           root=0)
        data_chunked = data_chunked_flat.reshape(-1, num_features)
        self.logger.info(f"Received data_chunked ({data_chunked.shape}):\n{data_chunked}")

        # Reduce and find average for cluster 1
        if self.rank == 0:
            self.logger.info(f"****Reduce!****")

        # Find avg for cluster 1 only
        data_chunked_clust_1 = data_chunked[cluster_assignments_chunked == 1]
        self.logger.info(f"Data for cluster 1 (shape: {data_chunked_clust_1.shape}:\n "
                         f"{data_chunked_clust_1}")
        # Find sum of each cluster
        size_cluster_1_chunked = data_chunked_clust_1.shape[0]
        if size_cluster_1_chunked > 0:
            sum_cluster_1_chunked = np.sum(data_chunked_clust_1, axis=0)
        else:
            sum_cluster_1_chunked = np.zeros_like(data_chunked[0])
        self.logger.info(f"Sum cluster 1 (shape: {sum_cluster_1_chunked.shape}: "
                         f"{sum_cluster_1_chunked}")
        # Reduce the internal sums to find total sum
        sum_cluster_1 = np.zeros_like(sum_cluster_1_chunked)
        self.comm.Reduce([sum_cluster_1_chunked, MPI.DOUBLE], [sum_cluster_1, MPI.DOUBLE],
                         op=MPI.SUM, root=0)

        self.logger.info(f"Chunked size: {size_cluster_1_chunked}")
        total_size = self.comm.reduce(size_cluster_1_chunked, op=MPI.SUM, root=0)
        if self.rank == 0:
            self.logger.info(f"Total size: {total_size}. Summed sums: {sum_cluster_1}")
            avg_cluster_1 = sum_cluster_1 / total_size
            self.logger.info(f"Average Cluster 1: {avg_cluster_1}")


if __name__ == '__main__':
    mpi_play = MPlayI()
    if sys.argv[1] == 'simple':
        mpi_play.simple()
    elif sys.argv[1] == 'broadcast':
        mpi_play.broadcast()
    elif sys.argv[1] == 'scatter_gather':
        mpi_play.scatter_gather()
    elif sys.argv[1] == 'all_gather':
        mpi_play.all_gather()
    elif sys.argv[1] == 'mpi_reduce':
        mpi_play.mpi_reduce()
    elif sys.argv[1] == 'mpi_all_reduce':
        mpi_play.mpi_all_reduce()
    elif sys.argv[1] == 'count_lines':
        mpi_play.count_lines()
    elif sys.argv[1] == 'reduce_complex':
        mpi_play.reduce_complex()
