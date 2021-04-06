"""Top-level package for playground."""

from playground.fancy_logger import ColorizedLogger
from playground.timing_tools import timeit
from playground.profiling_funcs import profileit
from playground.configuration import Configuration
from playground.Benchmarking import BenchTests, run_math_calc_test, run_fill_and_empty_list_test
from playground.MessagePassing import MPlayI
from playground.KMeans import KMeansRunner
from playground.numba_play import NumbaPlay

__author__ = "drkostas"
__email__ = "georgiou.kostas94@gmail.com"
__version__ = "0.1.0"
