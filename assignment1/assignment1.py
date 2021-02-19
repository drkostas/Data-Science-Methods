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
extra_ch_logger = ColorizedLog(logging.getLogger('ExtraMain'), 'yellow')
extra_sub_ch_logger = ColorizedLog(logging.getLogger('ExtraSub'), 'cyan')


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
                conf_type: required
    """

    p1_logger.info("Starting Problem 1..")
    conf_props = conf['properties']
    # Generate iterable from `x_min` to `x_max`
    xs = range(conf_props["x_min"], conf_props["x_max"] + 1)
    p1_logger.info(f"Will call `my_pid` for x in {tuple(xs)}")
    # Call my_pid() using pool.map()
    with multiprocessing.Pool(processes=conf_props['pool_size']) as pool:
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map
        pool.map(func=my_pid,
                 iterable=xs,
                 chunksize=conf_props['chunk_size'])


# Call the timeit wrapper from main.py and pass it a custom string to print
# It already supports string formatting for func_name`, `args`, and `duration`
# To reference the first positional argument of the function I wrap, I can use {0}
@timeit(custom_print='N={0}: Calculation of pi took: {duration:2.5f} sec(s) total')
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
    p2_logger.info(f"N={N}: Pi({N}) = {calced_pi} (Real is {real_pi}, difference is {pi_diff})")


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
                conf_type: required

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
    num_terms_range = tuple(num_terms_range)  # I only do this because I want to print them
    # Create the iterable of arguments to be passed to py_pi using pool.starmap()
    # repeat() propagates the same `real_pi` value as many times as necessary
    # to zip it with num_terms_range
    args = zip(num_terms_range, repeat(real_pi))
    p2_logger.info(f"Will call `py_pi` for N in {tuple(num_terms_range)}")
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
                conf_type: required
    """

    p3_logger.info("Starting Problem 3..")
    conf_props = conf['properties']
    p3_logger.info(f"Will call `py_pi_better` for N in {tuple(conf_props['num_terms'])}")
    # Run the pi calculation once for each number of terms requested in the yml
    for num_term in conf_props["num_terms"]:
        # Split the work of `num_terms` into `pool_size` number of parts
        step = ceil(num_term / conf_props["pool_size"])
        i_start = range(1, num_term, step)
        i_stop = map(lambda el: el + step - 1 if el + step - 1 <= num_term else num_term, i_start)
        # Zip N with the i_start and i_stop iterables. Propagate the same N value using repeat
        args = zip(repeat(num_term), i_start, i_stop)
        # Call py_pi_better() using pool.starmap() (starmap accepts iterable with multiple arguments)
        with multiprocessing.Pool(processes=conf_props['pool_size']) as pool:
            # timeit can be used as a context manager too. Pass it a custom string and count the
            # total time to calculate pi
            custom_string = f'N={num_term}: Parallel calculation of pi took: ' + \
                            '{duration:2.5f} sec(s) total'
            with timeit(custom_print=custom_string):
                # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map
                pi_chunks = pool.starmap(func=py_pi_better,
                                         iterable=args,
                                         chunksize=conf_props['chunk_size'])
                calced_pi = sum(pi_chunks)
        real_pi = np.pi
        pi_diff = abs(real_pi - calced_pi)
        p3_logger.info(f"N={num_term}: Pi({num_term}) = {calced_pi} (Real is {real_pi}, "
                       f"difference is {pi_diff})")


def extra_1(conf_props: Dict):
    """ Extra Challenge 1 solution

    Parameters:
         conf_props: The config loaded from the yml file
            Example:
                pool_sizes:
                  - 2
                  - 4
                  - 8
                  - 16
                  - 32
                  - 64
                chunk_size: 1
                num_term: 500000
    """

    num_term = int(conf_props["num_term"])
    extra_ch_logger.info("Will call `py_pi_better` for pool_size in "
                         f"{tuple(conf_props['pool_sizes'])}")
    for pool_size in conf_props["pool_sizes"]:
        pool_size = int(pool_size)
        # Slightly modified code from problem 3
        extra_sub_ch_logger.info(f"Pool Size={pool_size}: Calling workers for N={num_term}")
        step = ceil(num_term / pool_size)
        i_start = range(1, num_term, step)
        i_stop = map(lambda el: el + step - 1 if el + step - 1 <= num_term else num_term, i_start)
        args = zip(repeat(num_term), i_start, i_stop)
        with multiprocessing.Pool(processes=pool_size) as pool:
            custom_string = f'Pool Size={pool_size}: Calculation of pi for N={num_term} took: ' + \
                            '{duration:2.5f} sec(s) total'
            with timeit(custom_print=custom_string):
                pi_chunks = pool.starmap(func=py_pi_better,
                                         iterable=args,
                                         chunksize=conf_props['chunk_size'])
                calced_pi = sum(pi_chunks)
        real_pi = np.pi
        pi_diff = abs(real_pi - calced_pi)
        extra_sub_ch_logger.info(f"Pool Size={pool_size}: Pi({num_term}) = {calced_pi} "
                                 f"(Real is {real_pi}, difference is {pi_diff})")


def py_pi_better_with_queue(N: int, i_start: int, i_stop: int,
                            m_queue: multiprocessing.Queue) -> None:
    """ Extra Challenge 2 function to be called using pool.call_async

    Parameters:
        N: Number of terms to be used to calculate pi
        i_start: The starting index of the sum used in pi's approximation
        i_stop: The ending index of the sum used in pi's approximation
        m_queue: A queue in which to append the results
    """

    # Construct the equation
    first_term = (4 / N)
    list_to_be_summed = [1 / (1 + ((i - 0.5) / N) ** 2) for i in range(i_start, i_stop + 1)]
    second_term = sum(list_to_be_summed)
    cacled_pi = first_term * second_term
    m_queue.put(cacled_pi)


def extra_2(conf_props: Dict):
    """ Extra Challenge 2 solution

    Parameters:
         conf_props: The config loaded from the yml file
            Example:
    """

    num_term = int(conf_props["num_term"])

    extra_ch_logger.info("Will call `py_pi_better_with_queue` for pool_size in "
                         f"{tuple(conf_props['pool_sizes'])}")
    for pool_size in conf_props["pool_sizes"]:
        pool_size = int(pool_size)
        # Create a multiprocessing manager and a Queue
        multi_manager = multiprocessing.Manager()
        multi_queue = multi_manager.Queue()
        # Slightly modified code from problem 3
        extra_sub_ch_logger.info(f"Pool Size={pool_size}: Calling Async workers for N={num_term}")
        step = ceil(num_term / pool_size)
        i_start = range(1, num_term, step)
        i_stop = map(lambda el: el + step - 1 if el + step - 1 <= num_term else num_term, i_start)
        args = zip(repeat(num_term), i_start, i_stop, repeat(multi_queue))
        # Setup manually a Pool
        custom_string = f'Pool Size={pool_size}: Calculation of pi for N={num_term} and  ' + \
                        'took: {duration:2.5f} sec(s) total'
        with timeit(custom_print=custom_string):
            pool = multiprocessing.Pool(processes=pool_size)
            pool.starmap_async(func=py_pi_better_with_queue, iterable=args)
            pi_chunks = []
            while len(pi_chunks) < pool_size:
                pi_chunks.append(multi_queue.get())
            calced_pi = sum(pi_chunks)
        real_pi = np.pi
        pi_diff = abs(real_pi - calced_pi)
        extra_sub_ch_logger.info(f"Pool Size={pool_size}: Pi({num_term}) = {calced_pi}"
                                 f"(Real is {real_pi}, difference is {pi_diff})")
        pool.close()
        pool.join()


def extra_challenges(conf: Dict) -> None:
    """ Extra Challenges solution

    Parameters:
         conf: The config loaded from the yml file
            Example:
    """

    extra_ch_logger.info("Starting the Extra Challenges..")
    extra_ch_logger.info("1: Test performance for different pool sizes.")
    extra_1(conf['properties']["ch1"])
    extra_ch_logger.info("2: Collect results from workers using a queue (using starmap_async).")
    extra_2(conf['properties']["ch2"])


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
    main_logger.info(f"{' Required Problems ':-^{100}}")
    # Load the configuration
    conf = Configuration(config_src=args.config_file)
    # Start the problems defined in the configuration
    check_required = lambda conf_type, tag: ((conf_type == 'required' or tag != 'required_only')
                                             and conf_type != 'disabled')
    # For each problem present in the config file, call the appropriate function
    if 'problem1' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='problem1'):
            if check_required(bench_conf['type'], conf.tag):
                problem1(bench_conf)
    if 'problem2' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='problem2'):
            if check_required(bench_conf['type'], conf.tag):
                problem2(bench_conf)
    if 'problem3' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='problem3'):
            if check_required(bench_conf['type'], conf.tag):
                problem3(bench_conf)
    # Run the extra challenges if the tag of the conf is not set as "required_only"
    main_logger.info(f"{' Optional Problems ':-^{100}}")
    if 'extra_challenges' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='extra_challenges'):
            if check_required(bench_conf['type'], conf.tag):
                extra_challenges(bench_conf)
    main_logger.info("Assignment 1 Finished")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
