import logging
import traceback
import os
from typing import Dict
import multiprocessing
import numpy as np

from playground.main import _setup_log, _argparser, timeit
from playground.fancy_log.colorized_log import ColorizedLog
from playground.configuration.configuration import Configuration

main_logger = ColorizedLog(logging.getLogger('Main'), 'yellow')
p1_logger = ColorizedLog(logging.getLogger('Problem1'), 'blue')
p2_logger = ColorizedLog(logging.getLogger('Problem2'), 'green')
p3_logger = ColorizedLog(logging.getLogger('Problem3'), 'magenta')


# @timeit
def my_pid(x: int) -> None:
    """ Problem 1's function to be called using pool.map

    Parameters:
        x: the id of the worker
    """

    pid = os.getpid()
    p1_logger.info(f"Hi, I’m worker {x} (with {pid})")


def problem1(conf: Dict) -> None:
    """ Problem 1 solution

    Parameters:
         conf: The config loaded from the yml file
    """

    p1_logger.info("Starting Problem 1..")
    conf_props = conf['properties']
    # Generate iterable with pool_size random number in the range specified ([0-9])
    xs = range(conf_props["x_min"], conf_props["x_max"] + 1)
    # Call my_pid() using pool.map()
    with multiprocessing.Pool(processes=conf_props['pool_size']) as pool:
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map
        pool.map(func=my_pid,
                 iterable=xs,
                 chunksize=conf_props['chunk_size'])


@timeit
def py_pi(N: int, real_pi: float) -> None:
    """ Problem 2's function to be called using pool.map

    Parameters:
        N: Number of terms to be used to calculate pi
        real_pi: Correct value of pi
    """

    first_term = (4 / N)
    list_to_be_summed = [1 / (1 + ((i - 0.5) / N) ** 2) for i in range(1, N + 1)]
    second_term = sum(list_to_be_summed)
    calced_pi = first_term * second_term
    pi_diff = abs(real_pi - calced_pi)
    p2_logger.info(f"Pi({N}) = {calced_pi} (Real is {real_pi}, difference is {pi_diff})")


def problem2(conf: Dict) -> None:
    """ Problem 2 solution

    Parameters:
         conf: The config loaded from the yml file
    """

    p2_logger.info("Starting Problem 2..")
    real_pi = np.pi
    p2_logger.info("Expected pi value: %s" % real_pi)
    conf_props = conf['properties']
    # Generate iterable with number of terms to be used
    num_terms_range = []
    current_num_term = conf_props["num_terms_min"]
    while current_num_term < conf_props["num_terms_max"]:
        num_terms_range.append(current_num_term)
        current_num_term *= conf_props["num_terms_step"]
    args = [(num_terms, real_pi) for num_terms in num_terms_range]
    # Call py_pi() using pool.starmap() (starmap accepts iterable with multiple arguments)
    with multiprocessing.Pool(processes=conf_props['pool_size']) as pool:
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map
        pool.starmap(func=py_pi,
                     iterable=args,
                     chunksize=conf_props['chunk_size'])


@timeit
def main():
    """ This is the main function of assignment.py

    Example:
        python assignment1/assignment.py \
            -c ../confs/assignment1.yml \
            -l ../logs/assignment.log
    """

    # Initialize
    args = _argparser()
    _setup_log(args.log, args.debug)
    main_logger.info("Starting Assignment 1")
    # Load the configuration
    conf = Configuration(config_src=args.config_file)
    # Start
    if 'problem1' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='problem1'):
            problem1(bench_conf)
    if 'problem2' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='problem2'):
            problem2(bench_conf)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
