import argparse
import logging
import os
import sys
import traceback
from typing import Dict

from playground import ColorizedLogger, timeit, Configuration
from playground import run_math_calc_test, run_fill_and_empty_list_test

logger = ColorizedLogger(logger_name='Main', color='yellow')


def get_args() -> argparse.Namespace:
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
    optional_args.add_argument('--local', action='store_true',
                               help='Specifies that the script is running locally')
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
        cmd = 'mpirun -n {nprocs} {python} {file} {num_clusters} {type}' \
            .format(nprocs=nprocs,
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


def run_cprofile(conf: Dict, log_path: str) -> None:
    """ Runs the CProfile tests for the specified configuration. """

    config = conf['properties']
    func_to_run = config['func_to_run']
    prof_type = config['profiling']
    logger.info(f"Invoking profiling_play.py with func_to_run=`{func_to_run}`")
    logger.nl()
    sys_path = os.path.dirname(os.path.realpath(__file__))
    run_file_path = os.path.join(sys_path, 'profiling_funcs', 'profiling_play.py')
    # if run_type == 'mpi':
    #     nprocs = config['nprocs']
    #     cmd = 'mpirun -n {nprocs} {python} {file} {num_clusters} {type}' \
    #         .format(nprocs=nprocs,
    #                 python=sys.executable,
    #                 file=run_file_path,
    #                 type=run_type,
    #                 num_clusters=num_clusters)

    # elif run_type in ('simple', 'vectorized', 'distributed'):
    cmd = f'{sys.executable} {run_file_path} -f {func_to_run} -p {prof_type} -l {log_path}'
    # else:
    #     raise Exception(f'Argument {run_type} not recognized!')
    with timeit(custom_print=f'Running Profiling {func_to_run} took' +
                             ' {duration:2.5f} sec(s)'):
        os.system(cmd)


@timeit()
def main():
    """This is the main function of main.py

    Example: python playground/main.py -m run_mode_1

        -c confs/template_conf.yml -l logs/output.log
    """

    # Initializing
    args = get_args()
    log_path = os.path.abspath(args.log)
    ColorizedLogger.setup_logger(log_path, args.debug, clear_log=True)
    # Load the configuration
    conf = Configuration(config_src=args.config_file)
    # Start
    check_required = lambda conf_type, conf_enabled, tag: \
        ((conf_type == 'required' or tag != 'required_only') and conf_enabled)
    if 'bench' in conf.config_keys:
        for sub_config in conf.get_config(config_name='bench'):
            if check_required(sub_config['type'], sub_config['enabled'], conf.tag):
                run_bench(sub_config)
    if 'mpi' in conf.config_keys:
        for sub_config in conf.get_config(config_name='mpi'):
            if check_required(sub_config['type'], sub_config['enabled'], conf.tag):
                run_mpi(sub_config)
    if 'kmeans' in conf.config_keys:
        for sub_config in conf.get_config(config_name='kmeans'):
            if check_required(sub_config['type'], sub_config['enabled'], conf.tag):
                run_kmeans(sub_config)
    if 'cprofile' in conf.config_keys:
        for sub_config in conf.get_config(config_name='cprofile'):
            if check_required(sub_config['type'], sub_config['enabled'], conf.tag):
                run_cprofile(sub_config, log_path=log_path)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
