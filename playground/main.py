import traceback
import logging
import argparse
import os
import sys
import time
from typing import Dict

from playground.fancy_log.colorized_log import ColorizedLog
from playground.configuration.configuration import Configuration
from playground.Benchmarking.parallel_bench import run_math_calc_test, run_fill_and_empty_list_test

logger = ColorizedLog(logging.getLogger('Main'), 'yellow')


def timeit(method: object) -> object:
    """Decorator for counting the execution times of functions

    Args:
        method (object):
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            logger.info('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def _setup_log(log_path: str = '../logs/default.log', debug: bool = False) -> None:
    """Set the parameteres of the logger

    Args:
        log_path (str): The path the log file is going to be saved
        debug (bool): Whether to print debug messages or not
    """
    log_path = log_path.split(os.sep)
    if len(log_path) > 1:

        try:
            os.makedirs((os.sep.join(log_path[:-1])))
        except FileExistsError:
            pass
    log_filename = os.sep.join(log_path)
    # noinspection PyArgumentList
    logging.basicConfig(level=logging.INFO if debug is not True else logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(log_filename),
                            logging.StreamHandler()
                        ]
                        )


def _argparser() -> argparse.Namespace:
    """Setup the argument parser

    Returns:
        argparse.Namespace:
    """
    parser = argparse.ArgumentParser(
        description='A playground repo for the DSE-512 course..',
        add_help=False)
    # Required Args
    required_args = parser.add_argument_group('Required Arguments')
    config_file_params = {
        'type': argparse.FileType('r'),
        'required': True,
        'help': "The configuration yml file"
    }
    required_args.add_argument('-c', '--config-file', **config_file_params)
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    # optional_args.add_argument('-m', '--run-mode', choices=['run_mode_1', 'run_mode_2', 'run_mode_3'],
    #                            default='run_mode_1',
    #                            help='Description of the run modes')
    optional_args.add_argument('-l', '--log',
                               default='logs/default.log',
                               help="Name of the output log file")
    optional_args.add_argument('-d', '--debug', action='store_true',
                               help='Enables the debug log messages')
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


def run_bench(conf: Dict) -> None:
    """ Runs the bench tests for the specified configuration. """

    config = conf['config']
    type = conf['type']
    if type == 'math_calc':
        logger.info("Starting math_calc")
        run_math_calc_test(test_proc_threads=config['num_threads_processes'],
                           max_float=config['max_float'],
                           num_loops=config['num_loops'])
    elif type == 'fill_and_empty_list':
        logger.info("Starting fill_and_empty_list")
        run_fill_and_empty_list_test(test_proc_threads=config['num_threads_processes'],
                                     max_float=config['max_float'],
                                     num_loops=config['num_loops'])
    else:
        raise Exception("Config type not recognized!")


def run_mpi(conf: Dict) -> None:
    """ Runs the MPI tests for the specified configuration. """

    config = conf['config']
    type = conf['type']
    logger.info("Invoking play.py with type=`%s`" % type)
    sys_path = os.path.dirname(os.path.realpath(__file__))
    run_file_path = os.path.join(sys_path, 'MessagePassing', 'play.py')
    cmd = 'mpirun -n {nprocs} {python} {file} {type}'.format(nprocs=config['nprocs'],
                                                             python=sys.executable,
                                                             file=run_file_path,
                                                             type=type)
    os.system(cmd)


@timeit
def main():
    """This is the main function of main.py

    Example: python playground/main.py -m run_mode_1

        -c confs/template_conf.yml -l logs/output.log
    """

    # Initializing
    args = _argparser()
    _setup_log(args.log, args.debug)
    # Load the configuration
    conf = Configuration(config_src=args.config_file)

    # Start
    if 'bench' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='bench'):
            run_bench(bench_conf)

    if 'mpi' in conf.config_keys:
        for mpi_conf in conf.get_config(config_name='mpi'):
            run_mpi(mpi_conf)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
