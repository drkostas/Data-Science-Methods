import logging
import traceback
from typing import List, Dict

from playground.main import get_args
from playground import ColorizedLogger, Configuration, timeit
from kmeans import KMeansRunner

# Create loggers with different colors to use in each problem
main_logger = ColorizedLogger('Main', 'yellow')


def prepare_for_run(name: str, conf: Dict):
    conf_props = conf['properties']
    num_clusters = conf_props['num_clusters']
    dataset = conf_props['dataset']
    dataset_name = 'tcga' if dataset != 'iris' else dataset
    main_logger.info(f"Invoking {name} for {num_clusters} clusters and the {dataset_name} dataset")
    return num_clusters, dataset, dataset_name


def check_required(conf_type, conf_enabled, tag):
    return (conf_type == 'required' or tag != 'required_only') and conf_enabled


def run_serial(name: str, config_key: List[Dict], tag: str) -> None:
    """ Runs the KMeans serial version for the specified configuration. """
    # Initialize
    kmeans_runner = KMeansRunner(run_type=name)
    for bench_conf in config_key:
        if check_required(bench_conf['type'], bench_conf['enabled'], tag):
            # Extract the properties
            num_clusters, dataset, dataset_name = prepare_for_run(name, bench_conf)
            # Run K-Means
            kmeans_runner.run(num_clusters=num_clusters, dataset=dataset)


@timeit()
def main():
    """ This is the main function of assignment3.py

    Example:
        python assignment3/assignment3.py \
            -c ../confs/assignment3.yml \
            -l ../logs/assignment3.log
    """

    # Initialize
    args = get_args()
    ColorizedLogger.setup_logger(args.log, args.debug, True)
    main_logger.info("Starting Assignment 3")
    # Load the configuration
    conf = Configuration(config_src=args.config_file)
    # Start the problems defined in the configuration
    # For each problem present in the config file, call the appropriate function
    for config_key in conf.config_keys:
        run_serial(name=config_key, config_key=conf.get_config(config_name=config_key), tag=conf.tag)

    main_logger.info("Assignment 3 Finished")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
