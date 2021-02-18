import logging
import traceback
import os
from typing import Dict
from math import ceil
from itertools import repeat, takewhile

import multiprocessing
import numpy as np

from playground.main import setup_log, argparser, timeit
from playground.fancy_log.colorized_log import ColorizedLog
from playground.configuration.configuration import Configuration

# Create loggers with different colors to use in each problem
main_logger = ColorizedLog(logging.getLogger('Main'), 'yellow')
p1_logger = ColorizedLog(logging.getLogger('Problem1'), 'blue')
p2_logger = ColorizedLog(logging.getLogger('Problem2'), 'green')
p3_logger = ColorizedLog(logging.getLogger('Problem3'), 'magenta')


def my_pid(x: int) -> None:
    """ Problem 1 function to be called using pool.map

    Parameters:
        x: the id of the worker
    """

    pid = os.getpid()
    p1_logger.info(f"Hi, I’m worker {x} (with {pid})")


def problem1(conf: Dict) -> None:
    """ Problem 1 solution

    Parameters:
         conf: The config loaded from the yml file
            Example:
                properties:
                  pool_size: 4
                  chunk_size: 1
                  x_min: 0
                  x_max: 9
                type: required
    """

    p1_logger.info("Starting Problem 1..")
    conf_props = conf['properties']
    # Generate iterable from `x_min` to `x_max`
    xs = range(conf_props["x_min"], conf_props["x_max"] + 1)
    # Call my_pid() using pool.map()
    with multiprocessing.Pool(processes=conf_props['pool_size']) as pool:
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map
        pool.map(func=my_pid,
                 iterable=xs,
                 chunksize=conf_props['chunk_size'])


# Call the timeit wrapper from main.py and pass it a custom string to print
# It already supports string formatting for func_name`, `args`, and `duration`
# To reference the first positional argument of the function I wrap, I can use {0}
@timeit(custom_print='Calculation of pi for N={0} took: {duration:2.5f} sec(s) total')
def py_pi(N: int, real_pi: float) -> None:
    """ Problem 2 function to be called using pool.starmap

    Parameters:
        N: Number of terms to be used to calculate pi
        real_pi: Correct value of pi
    """

    # Construct the equation
    first_term = (4 / N)
    list_to_be_summed = [1 / (1 + ((i - 0.5) / N) ** 2) for i in range(1, N + 1)]
    second_term = sum(list_to_be_summed)
    calced_pi = first_term * second_term
    # Calculate the absolute difference between the calculated pi and real value of pi
    pi_diff = abs(real_pi - calced_pi)
    p2_logger.info(f"Pi({N}) = {calced_pi} (Real is {real_pi}, difference is {pi_diff})")


def problem2(conf: Dict) -> None:
    """ Problem 2 solution

    Parameters:
         conf: The config loaded from the yml file
            Example:
                properties:
                  pool_size: 4
                  chunk_size: 1
                  num_terms_min: 10
                  num_terms_step: 5
                  num_terms_max: 3906250
                type: required

    """

    p2_logger.info("Starting Problem 2..")
    real_pi = np.pi
    conf_props = conf['properties']
    # Generate iterable with number of terms to be used
    # Lambda function that multiplies each element by 5 in the power of the element's index
    # e.g. [(10, 0), (10, 1), (10, 2)] -> [10*5^0, 10*5^1, 10*5^2]
    # the first ind_and_num would be (10, 0)
    multiplied_series = lambda ind_and_num: (ind_and_num[1] *
                                             (conf_props["num_terms_step"] ** (ind_and_num[0])))
    # Map an infinite enumerated iterable of (index, 10) to the lambda function
    # Stop when the series exceeds 3906250
    num_terms_range = takewhile(lambda series_el: series_el <= conf_props["num_terms_max"],
                                map(multiplied_series, enumerate(repeat(conf_props["num_terms_min"]))))
    # Create the iterable of arguments to be passed to py_pi using pool.starmap()
    # repeat() propagates the same `real_pi` value as many times as necessary
    # to zip it with num_terms_range
    args = zip(num_terms_range, repeat(real_pi))
    # Call py_pi() using pool.starmap() (starmap accepts iterable with multiple arguments)
    with multiprocessing.Pool(processes=conf_props['pool_size']) as pool:
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map
        pool.starmap(func=py_pi,
                     iterable=args,
                     chunksize=conf_props['chunk_size'])


def py_pi_better(N: int, i_start: int, i_stop: int) -> float:
    """ Problem 3 function to be called using pool.starmap

    Parameters:
        N: Number of terms to be used to calculate pi
        i_start: The starting index of the sum used in pi's approximation
        i_stop: The ending index of the sum used in pi's approximation
    Returns:
        partial_calced_pi: The part of pi's approximation calculated
    """

    # Construct the equation
    first_term = (4 / N)
    list_to_be_summed = [1 / (1 + ((i - 0.5) / N) ** 2) for i in range(i_start, i_stop + 1)]
    second_term = sum(list_to_be_summed)
    cacled_pi = first_term * second_term
    return cacled_pi


def problem3(conf: Dict) -> None:
    """ Problem 3 solution

    Parameters:
         conf: The config loaded from the yml file
            Example:
                properties:
                  pool_size: 4
                  chunk_size: 1
                  num_terms:
                    - 100
                    - 500
                    - 1000
                    - 2000
                    - 10000
                    - 50000
                type: required
    """

    p3_logger.info("Starting Problem 3..")
    conf_props = conf['properties']
    # Run the pi calculation once for each number of terms requested in the yml
    for num_term in conf_props["num_terms"]:
        p3_logger.info(f"Calling workers for N={num_term}")
        # Split the work of `num_terms` into `pool_size` number of parts
        step = ceil(num_term / conf_props["pool_size"])
        i_start = range(1, num_term, step)
        i_stop = map(lambda el: el + step - 1 if el + step - 1 <= num_term else num_term,
                     i_start)
        # Zip N with the i_start and i_stop iterables. Propagate the same N value using repeat
        args = zip(repeat(num_term), i_start, i_stop)
        # Call py_pi_better() using pool.starmap() (starmap accepts iterable with multiple arguments)
        with multiprocessing.Pool(processes=conf_props['pool_size']) as pool:
            # timeit can be used as a context manager too. Pass it a custom string and count the
            # total time to calculate pi
            custom_string = f'Calculation of pi for N={num_term} took: ' + \
                            '{duration:2.5f} sec(s) total'
            with timeit(custom_print=custom_string):
                # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map
                calced_pi = pool.starmap(func=py_pi_better,
                                         iterable=args,
                                         chunksize=conf_props['chunk_size'])


@timeit()
def main():
    """ This is the main function of assignment.py

    Example:
        python assignment1/assignment.py \
            -c ../confs/assignment1.yml \
            -l ../logs/assignment.log
    """

    # Initialize
    args = argparser()
    setup_log(args.log, args.debug)
    main_logger.info("Starting Assignment 1")
    # Load the configuration
    conf = Configuration(config_src=args.config_file)
    # Start the problems defined in the configuration
    if 'problem1' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='problem1'):
            problem1(bench_conf)
    if 'problem2' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='problem2'):
            problem2(bench_conf)
    if 'problem3' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='problem3'):
            problem3(bench_conf)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
