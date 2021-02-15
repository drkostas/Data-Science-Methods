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

        self.logger = ColorizedLog(logging.getLogger('MPI'), self.colors[self.rank])

    @staticmethod
    def _mpi_log_setup():
        sys_path = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(sys_path, '..', '..', 'logs', 'mpi.log')
        _setup_log(log_path=log_path)

    def simple(self):
        self.logger.info(f"Hello from rank {self.rank} of size {self.size}")
        # Wait for everyone to sync up
        self.comm.Barrier()


        print(f"Rank {rank} before broadcast has {x}")
        comm.Bcast([x, MPI.DOUBLE])
        print(f"Rank {rank} after broadcast has {x}\n")


if __name__ == '__main__':
    mpi_play = MPlayI()
    if sys.argv[1] == 'simple':
        mpi_play.simple()
