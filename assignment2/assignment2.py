import logging
import traceback
import os
import sys
from typing import Dict, List

from playground.main import setup_log, argparser, timeit
from playground.fancy_log.colorized_log import ColorizedLog
from playground.configuration.configuration import Configuration

# Create loggers with different colors to use in each problem
main_logger = ColorizedLog(logging.getLogger('Main'), 'yellow')
simp_logger = ColorizedLog(logging.getLogger('KMeans Simple'), 'cyan')
vect_logger = ColorizedLog(logging.getLogger('KMeans Vectorized'), 'blue')
distr_logger = ColorizedLog(logging.getLogger('KMeans Distributed'), 'green')


def run_simple(conf: Dict) -> None:
    """ Runs the KMeans simple version for the specified configuration. """

    simp_logger.info("KMeans Simple..")
    conf_props = conf['properties']
    num_clusters = conf_props['num_clusters']
    simp_logger.info(f"Invoking kmeans simple with num_clusters=`{num_clusters}`")
    sys_path = os.path.dirname(os.path.realpath(__file__))
    run_file_path = os.path.join(sys_path, 'kmeans.py')

    cmd = '{python} {file} -k {num_clusters}'.format(python=sys.executable,
                                                     file=run_file_path,
                                                     num_clusters=num_clusters)
    with timeit(custom_print=f'Running KMeans simple for {num_clusters} clusters took' +
                             ' {duration:2.5f} sec(s)'):
        os.system(cmd)


def run_vectorized(conf: Dict, jacob_version: bool = False) -> None:
    """ Runs the KMeans vectorized version for the specified configuration. """

    vect_logger.info("KMeans Vectorized Jacob's version..") if jacob_version \
        else vect_logger.info("KMeans Vectorized..")
    conf_props = conf['properties']
    num_clusters = conf_props['num_clusters']
    if jacob_version:
        python_file_name = 'kmeans_vectorized_jacob.py'
    else:
        python_file_name = 'kmeans_vectorized.py'
    vect_logger.info(f"Invoking {python_file_name} with num_clusters=`{num_clusters}`")
    sys_path = os.path.dirname(os.path.realpath(__file__))
    if jacob_version:
        python_file_name = 'kmeans_vectorized_jacob.py'
    else:
        python_file_name = 'kmeans_vectorized.py'
    run_file_path = os.path.join(sys_path, python_file_name)
    cmd = '{python} {file} -k {num_clusters}'.format(python=sys.executable,
                                                     file=run_file_path,
                                                     num_clusters=num_clusters)
    with timeit(custom_print=f'Running {python_file_name} for {num_clusters} clusters took' +
                             ' {duration:2.5f} sec(s)'):
        os.system(cmd)


def run_distributed(conf: Dict, jacob_version: bool = False) -> None:
    """ Runs the KMeans distributed version for the specified configuration. """

    vect_logger.info("KMeans Distributed Jacob's version..") if jacob_version \
        else vect_logger.info("KMeans Distributed..")
    conf_props = conf['properties']
    num_clusters = conf_props['num_clusters']
    nprocs = conf_props['nprocs']
    if jacob_version:
        python_file_name = 'kmeans_distributed_jacob.py'
    else:
        python_file_name = 'kmeans_distributed.py'
    vect_logger.info(f"Invoking {python_file_name} with num_clusters=`{num_clusters}`")
    sys_path = os.path.dirname(os.path.realpath(__file__))
    if jacob_version:
        python_file_name = 'kmeans_distributed_jacob.py'
    else:
        python_file_name = 'kmeans_distributed.py'
    run_file_path = os.path.join(sys_path, python_file_name)
    cmd = 'mpirun -n {nprocs} {python} {file} -k {num_clusters}'.format(nprocs=nprocs,
                                                                        python=sys.executable,
                                                                        file=run_file_path,
                                                                        num_clusters=num_clusters)
    with timeit(custom_print=f'Running {python_file_name} for {num_clusters} clusters took' +
                             ' {duration:2.5f} sec(s)'):
        os.system(cmd)


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
    main_logger.info("Starting Assignment 2")
    # Load the configuration
    conf = Configuration(config_src=args.config_file)
    # Start the problems defined in the configuration
    main_logger.info(f"{' Required Problems ':-^{100}}")
    check_required = lambda conf_type, conf_enabled, tag: \
        ((conf_type == 'required' or tag != 'required_only') and conf_enabled)
    # For each problem present in the config file, call the appropriate function
    if 'simple' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='simple'):
            if check_required(bench_conf['type'], bench_conf['enabled'], conf.tag):
                run_simple(bench_conf)
    if 'vectorized_jacob' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='vectorized'):
            if check_required(bench_conf['type'], bench_conf['enabled'], conf.tag):
                run_vectorized(bench_conf, jacob_version=True)
    if 'vectorized' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='vectorized'):
            if check_required(bench_conf['type'], bench_conf['enabled'], conf.tag):
                run_vectorized(bench_conf)
    if 'distributed_jacob' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='distributed_jacob'):
            if check_required(bench_conf['type'], bench_conf['enabled'], conf.tag):
                run_distributed(bench_conf, jacob_version=True)
    if 'distributed' in conf.config_keys:
        for bench_conf in conf.get_config(config_name='distributed'):
            if check_required(bench_conf['type'], bench_conf['enabled'], conf.tag):
                run_distributed(bench_conf)
    main_logger.info("Assignment 2 Finished")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
