import os
import sys
import logging
import traceback
from typing import List, Dict

from playground.main import get_args
from playground import ColorizedLogger, Configuration, timeit
from cnn_runner import CnnRunner

# Create loggers with different colors to use in each problem
main_logger = ColorizedLogger('Main', 'yellow')


def check_required(conf_type, conf_enabled, tag):
    return (conf_type == 'required' or tag != 'required_only') and conf_enabled


def run_distributed(conf_props: Dict, log_name: str, local: bool = False) -> None:
    """ Runs the KMeans distributed version for the specified configuration. """

    if local:
        mpi_path = os.path.join('/usr', 'bin', 'mpirun')
    else:
        env_path = f'{os.sep}'.join(sys.executable.split(os.sep)[:-2])
        mpi_path = os.path.sep + os.path.join(env_path, 'bin', 'mpirun')

    # Construct the command
    sys_path = os.path.dirname(os.path.realpath(__file__))
    run_file_path = os.path.join(sys_path, 'cnn_runner.py')
    for num_processes in conf_props['num_processes']:
        cmd = f"{mpi_path} -n {num_processes} {sys.executable} {run_file_path} " \
              f"--dataset-name {conf_props['dataset']['name']} " \
              f"--dataset-path {conf_props['dataset']['save_path']} " \
              f"-ep {conf_props['epochs']} " \
              f"-btr {conf_props['batch_size_train']} " \
              f"-bte {conf_props['batch_size_test']} " \
              f"-lr {conf_props['learning_rate']} " \
              f"-mo {conf_props['momentum']} " \
              f"-l {log_name} " \
              f"--test-before-train"
        # Run
        os.system(cmd)


def run(run_type: str, config: List[Dict], tag: str, log_name: str, local: bool) -> None:
    """Calls the CNN Runner for the specified configuration. """

    data_parallel = (run_type == 'data_parallel')
    if data_parallel:
        for conf in config:
            if check_required(conf['type'], conf['enabled'], tag):
                conf_props = conf['properties']
                run_distributed(conf_props, log_name, local)
    else:
        for conf in config:
            if check_required(conf['type'], conf['enabled'], tag):
                conf_props = conf['properties']
                # Run
                cr = CnnRunner(dataset=conf_props['dataset'],
                               epochs=conf_props['epochs'],
                               batch_size_train=conf_props['batch_size_train'],
                               batch_size_test=conf_props['batch_size_test'],
                               learning_rate=conf_props['learning_rate'],
                               momentum=conf_props['momentum'],
                               test_before_train=conf_props['test_before_train'])
                for num_processes in conf_props['num_processes']:
                    cr.run(num_processes=num_processes)


@timeit()
def main():
    """ This is the main function of assignment4.py

    Example:
        python assignment4/assignment4.py \
            -c ../confs/assignment4.yml \
            -l ../logs/assignment4.log
    """

    # Initialize
    args = get_args()
    ColorizedLogger.setup_logger(args.log, args.debug, True)
    main_logger.info("Starting Assignment 4")
    # Load the configuration
    conf = Configuration(config_src=args.config_file)
    # Start the problems defined in the configuration
    # For each problem present in the config file, call the appropriate function
    for config_key in conf.config_keys:
        run(run_type=config_key,
            config=conf.get_config(config_name=config_key),
            tag=conf.tag,
            log_name=args.log,
            local=args.local)

    main_logger.info("Assignment 4 Finished")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
