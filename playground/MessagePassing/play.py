import sys
import os
import logging
from mpi4py import MPI
import numpy as np

from playground.fancy_log.colorized_log import ColorizedLog
from playground.main import _setup_log


class MPlayI:
    colors = {
        0: 'blue',
        1: 'green',
        2: 'magenta',
        3: 'cyan'
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
        _setup_log(log_path=log_path)

    def simple(self):
        self.logger.info(f"Hello from rank {self.rank} of size {self.size}")
        # Wait for everyone to sync up
        self.comm.Barrier()

    def numpy_simple(self):

        if self.rank == 0:
            x = np.random.randn(4)
        else:
            x = np.empty(4, dtype=np.float64)

        self.logger.info(f"Rank {self.rank} before broadcast has {x}")
        self.comm.Bcast([x, MPI.DOUBLE])
        self.logger.info(f"Rank {self.rank} after broadcast has {x}")


if __name__ == '__main__':
    mpi_play = MPlayI()
    if sys.argv[1] == 'simple':
        mpi_play.simple()
    elif sys.argv[1] == 'numpy_simple':
        mpi_play.numpy_simple()
