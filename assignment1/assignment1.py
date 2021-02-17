import logging
import traceback
import os
from random import sample
from typing import Dict
import multiprocessing

from playground.main import _setup_log, _argparser, timeit
from playground.fancy_log.colorized_log import ColorizedLog
from playground.configuration.configuration import Configuration

logger = ColorizedLog(logging.getLogger('Assignment'), 'yellow')


# @timeit
def my_pid(x: int) -> None:
    pid = os.getpid()
    logger.info(f"Hi, I’m worker {x} (with {pid})")


def problem1(conf: Dict) -> None:
    conf_props = conf['properties']
    # Generate list with pool_size random number in the range specified ([0-9])
    xs = sample(range(conf_props["x_min"], conf_props["x_max"]+1), conf_props['pool_size'])

    with multiprocessing.Pool(processes=conf_props['pool_size']) as pool:
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map
        pool.map(my_pid, xs, conf_props['chunk_size'])


@timeit
def main():
    """This is the main function of assignment.py

    Example:
        python assignment1/assignment.py \
            -c ../confs/assignment1.yml \
            -l ../logs/assignment.log
    """

    # Initialize
    args = _argparser()
    _setup_log(args.log, args.debug)
    # Load the configuration
    conf = Configuration(config_src=args.config_file)
    # Start
    if 'problem1' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='problem1'):
            problem1(bench_conf)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
