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


def run(config: List[Dict], tag: str) -> None:
    """ Runs the KMeans serial version for the specified configuration. """
    # Initialize
    for conf in config:
        if check_required(conf['type'], conf['enabled'], tag):
            conf_props = conf['properties']
            # Run
            cr = CnnRunner(dataset=conf_props['dataset'],
                           epochs=conf_props['epochs'],
                           batch_size=conf_props['batch_size'],
                           learning_rate=conf_props['learning_rate'],
                           momentum=conf_props['momentum'])
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
        run(config=conf.get_config(config_name=config_key), tag=conf.tag)
        pass

    main_logger.info("Assignment 4 Finished")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
