from typing import Dict
import random
from numba import jit, njit, prange, stencil
import numpy as np
from playground import ColorizedLogger, timeit
import math


class NumbaPlay:
    logger: ColorizedLogger
    conf: Dict

    def __init__(self, conf):
        self.logger = ColorizedLogger(f'NumbaPlay', 'blue')
        self.logger.info(f"Initialized NumbaPlay..")
        self.conf = conf

    @staticmethod
    @jit(nopython=True)
    def pythagorean_theorem(x: int, y: int):
        return math.sqrt(x ** 2 + y ** 2)

    @staticmethod
    def pythagorus(x, y):
        return math.sqrt(x ** 2 + y ** 2)

    def pythagorean_test(self):
        self.logger.info(f"Starting pythagorean tests for x={self.conf['x']}, y={self.conf['x']}..")
        with timeit(custom_print='Numba pythagorean took {duration:.5f} sec(s)'):
            self.pythagorean_theorem(self.conf['x'], self.conf['y'])
        with timeit(custom_print='No Numba pythagorean took {duration:.5f} sec(s)'):
            self.pythagorus(self.conf['x'], self.conf['y'])

    @staticmethod
    @njit
    def monte_carlo_pi(nsamples):
        acc = 0
        for i in range(nsamples):
            x = random.random()
            y = random.random()
            if (x ** 2 + y ** 2) < 1.0:
                acc += 1
        return 4.0 * acc / nsamples

    def monte_carlo_pi_test(self):
        self.logger.info(f"Starting monte_carlo_pi tests for nsamples={self.conf['nsamples']}..")
        with timeit(custom_print='Numba monte_carlo_pi took {duration:.5f} sec(s)'):
            self.monte_carlo_pi(self.conf['nsamples'])

    @staticmethod
    @njit(parallel=True)
    def prange(A):
        s = 0
        # Without "parallel=True" in the jit-decorator
        # the prange statement is equivalent to range
        for i in prange(A.shape[0]):
            s += A[i]
        return s

    def prange_test(self):
        self.logger.info(f"Starting prange tests for A={self.conf['A']}..")
        with timeit(custom_print='Numba prange took {duration:.5f} sec(s)'):
            self.prange(np.arange(self.conf['A']))

    @staticmethod
    @jit(nopython=True, parallel=True)
    def logistic_regression(X, Y, w, iterations):
        for i in range(iterations):
            w -= np.dot(((1.0 /
                          (1.0 + np.exp(-Y * np.dot(X, w)))
                          - 1.0) * Y), X)
        return w

    def logistic_regression_test(self):
        self.logger.info(f"Starting logistic_regression tests for "
                         f"X={self.conf['x1']}, Y={self.conf['x2']}, "
                         f"w={self.conf['w']}, iterations={self.conf['iterations']}..")
        with timeit(custom_print='Numba logistic_regression took {duration:.5f} sec(s)'):
            self.logistic_regression(np.random.rand(self.conf['x1'], self.conf['x2']),
                                     np.random.rand(self.conf['x1']),
                                     np.zeros([self.conf['x2']]),
                                     self.conf['iterations'])
