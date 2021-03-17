import logging
import traceback
import os
import sys
from typing import Dict

from playground.main import setup_log, argparser, timeit
from playground.fancy_log.colorized_log import ColorizedLog
from playground.configuration.configuration import Configuration

# Create loggers with different colors to use in each problem
main_logger = ColorizedLog(logging.getLogger('Main'), 'yellow')


def prepare_for_run(name: str, conf: Dict):
    conf_props = conf['properties']
    num_clusters = conf_props['num_clusters']
    dataset = conf_props['dataset']
    python_file_name = 'kmeans.py'
    dataset_name = 'tcga' if dataset != 'iris' else dataset
    main_logger.info(f"Invoking {python_file_name}({name}) "
                     f"for {num_clusters} clusters and the {dataset_name} dataset")
    return python_file_name, num_clusters, dataset, dataset_name


def run_serial(name: str, conf: Dict, log_name: str) -> None:
    """ Runs the KMeans ser9ap version for the specified configuration. """

    # Extract the properties
    python_file_name, num_clusters, dataset, dataset_name = prepare_for_run(name, conf)
    # Construct the command
    sys_path = os.path.dirname(os.path.realpath(__file__))
    run_file_path = os.path.join(sys_path, python_file_name)
    cmd = f'{sys.executable} {run_file_path} -k {num_clusters} -d {dataset} -t {name} -l {log_name}'
    # Run
    os.system(cmd)


def run_distributed(name: str, conf: Dict, log_name: str, local: bool = False) -> None:
    """ Runs the KMeans distributed version for the specified configuration. """

    if local:
        mpi_path = os.path.join('usr', 'bin', 'mpirun')
    else:
        env_path = f'{os.sep}'.join(sys.executable.split(os.sep)[:-2])
        mpi_path = os.path.join(env_path, 'bin', 'mpirun')
    python_file_name, num_clusters, dataset, dataset_name = prepare_for_run(name, conf)
    nprocs = conf['properties']['nprocs']
    # Construct the command
    sys_path = os.path.dirname(os.path.realpath(__file__))
    run_file_path = os.path.join(sys_path, python_file_name)
    cmd = f'{mpi_path} -n {nprocs} {sys.executable} {run_file_path} -k {num_clusters} ' \
          f'-d {dataset} -t {name} -l {log_name}'
    # Run
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
    setup_log(args.log, args.debug, 'w')
    main_logger.info("Starting Assignment 2")
    # Load the configuration
    conf = Configuration(config_src=args.config_file)
    # Start the problems defined in the configuration
    check_required = lambda conf_type, conf_enabled, tag: \
        ((conf_type == 'required' or tag != 'required_only') and conf_enabled)
    # For each problem present in the config file, call the appropriate function
    for config_key in conf.config_keys:
        if 'distributed' in config_key:
            for bench_conf in conf.get_config(config_name=config_key):
                if check_required(bench_conf['type'], bench_conf['enabled'], conf.tag):
                    run_distributed(name=config_key, conf=bench_conf, log_name=args.log,
                                    local=args.local)
        else:
            for bench_conf in conf.get_config(config_name=config_key):
                if check_required(bench_conf['type'], bench_conf['enabled'], conf.tag):
                    run_serial(name=config_key, conf=bench_conf, log_name=args.log)

    main_logger.info("Assignment 2 Finished")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
