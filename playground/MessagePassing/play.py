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
        4: 'yellow'
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
