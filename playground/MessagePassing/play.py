import sys
import os
import logging
from typing import Dict
from mpi4py import MPI
import numpy as np

from playground.fancy_log.colorized_log import ColorizedLog
from playground.main import setup_log


class MPlayI:
    __slots__ = ('comm', 'rank', 'size', 'logger')
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
        self._mpi_log_setup()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size

        self.logger = ColorizedLog(logging.getLogger('MPI %s' % self.rank), self.colors[self.rank])

    @staticmethod
    def _mpi_log_setup():
        sys_path = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(sys_path, '..', '..', 'logs', 'mpi.log')
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
