import traceback
import logging
import argparse
import os
import sys
from time import time
from functools import wraps
from typing import Dict, Callable
from contextlib import ContextDecorator

from playground.fancy_log.colorized_log import ColorizedLog
from playground.configuration.configuration import Configuration
from playground.Benchmarking.parallel_bench import run_math_calc_test, run_fill_and_empty_list_test

logger = ColorizedLog(logging.getLogger('Main'), 'yellow')
time_logger = ColorizedLog(logging.getLogger('Timeit'), 'white')


class timeit(ContextDecorator):

    def __init__(self, **kwargs):
        """Decorator/ContextManager for counting the execution times of functions

        Args:
            custom_print: Custom print string which can be formatted using `func_name`, `args`,
                          and `duration`. Use {0}, {1}, .. to reference the first, second, ... argument
        """
        self.__dict__.update(kwargs)

    def __call__(self, func: Callable):
        """ This is called only when invoked as a decorator

        Args:
            method: The method to wrap
        """

        @wraps(func)
        def timed(*args, **kwargs):
            with self._recreate_cm():
                self.func_name = func.__name__
                self.args = args
                self.kwargs = kwargs
                self.all_args = (*args, *kwargs.values()) if kwargs != {} else args
                return func(*args, **kwargs)

        return timed

    def __enter__(self, *args, **kwargs):
        self.ts = time()
        return self

    def __exit__(self, type, value, traceback):
        self.te = time()
        total = self.te - self.ts
        if hasattr(self, 'func_name'):
            if not hasattr(self, 'custom_print'):
                print_string = 'Func: {func_name!r} with args: {args!r} took: {duration:2.5f} sec(s)'
            else:
                print_string = self.custom_print
            time_logger.info(print_string.format(*self.args, func_name=self.func_name,
                                                 args=self.all_args,
                                                 duration=total,
                                                 **self.kwargs))
        else:
            if not hasattr(self, 'custom_print'):
                print_string = 'Code block took: {duration:2.5f} sec(s)'
            else:
                print_string = self.custom_print
            if hasattr(self, 'file'):
                self.file.write(print_string.format(duration=total))
            if hasattr(self, 'skip'):
                if not self.skip:
                    time_logger.info(print_string.format(duration=total))
            else:
                time_logger.info(print_string.format(duration=total))


def setup_log(log_path: str = '../logs/default.log', debug: bool = False, mode: str = 'a') -> None:
    """Set the parameters of the logger

    Args:
        log_path (str): The path the log file is going to be saved
        debug (bool): Whether to print debug messages or not
        mode: (str): The write mode
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
                            logging.FileHandler(log_filename, mode=mode),
                            logging.StreamHandler()
                        ]
                        )


def argparser() -> argparse.Namespace:
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
    try:
        os.system(cmd)
    except Exception as e:
        logging.error(f"Command `{cmd}` failed:")
        raise e


def run_kmeans(conf: Dict) -> None:
    """ Runs the KMeans tests for the specified configuration. """

    config = conf['config']
    run_type = conf['type']
    num_clusters = config['num_clusters']
    logger.info(f"Invoking kmeans.py with type=`{run_type}` and num_clusters=`{num_clusters}`")
    sys_path = os.path.dirname(os.path.realpath(__file__))
    run_file_path = os.path.join(sys_path, 'KMeans', 'kmeans.py')
    if run_type == 'mpi':
        nprocs = config['nprocs']
        cmd = 'mpirun -n {nprocs} {python} {file} {num_clusters} {type}'.format(nprocs=nprocs,
                                                                                python=sys.executable,
                                                                                file=run_file_path,
                                                                                type=run_type,
                                                                                num_clusters=num_clusters)

    elif run_type in ('simple', 'vectorized', 'distributed'):
        cmd = '{python} {file} {num_clusters} {type}'.format(python=sys.executable,
                                                             file=run_file_path,
                                                             type=run_type,
                                                             num_clusters=num_clusters)
    else:
        raise Exception(f'Argument {run_type} not recognized!')
    with timeit(custom_print=f'Running KMeans {run_type} for {num_clusters} clusters took' +
                             ' {duration:2.5f} sec(s)'):
        os.system(cmd)


@timeit()
def main():
    """This is the main function of main.py

    Example: python playground/main.py -m run_mode_1

        -c confs/template_conf.yml -l logs/output.log
    """

    # Initializing
    args = argparser()
    setup_log(args.log, args.debug)
    # Load the configuration
    conf = Configuration(config_src=args.config_file)

    # Start
    if 'bench' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='bench'):
            if bench_conf['enabled']:
                run_bench(bench_conf)

    if 'mpi' in conf.config_keys:
        for mpi_conf in conf.get_config(config_name='mpi'):
            if mpi_conf['enabled']:
                run_mpi(mpi_conf)

    if 'kmeans' in conf.config_keys:
        for kmeans_conf in conf.get_config(config_name='kmeans'):
            if kmeans_conf['enabled']:
                run_kmeans(kmeans_conf)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
