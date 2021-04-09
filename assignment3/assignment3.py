import logging
import traceback
from typing import List, Dict

from playground.main import get_args
from playground import ColorizedLogger, Configuration, timeit
from kmeans import KMeansRunner
from kmeans_numba import run as run_with_numba

# Create loggers with different colors to use in each problem
main_logger = ColorizedLogger('Main', 'yellow')


def prepare_for_run(name: str, conf: Dict):
    conf_props = conf['properties']
    if 'numba' in conf_props:
        with_numba = conf_props['numba']
        name += '_numba'
    else:
        with_numba = False
    num_clusters = conf_props['num_clusters']
    dataset = conf_props['dataset']
    dataset_name = 'tcga' if dataset != 'iris' else dataset
    main_logger.info(f"Invoking {name} for {num_clusters} clusters and the {dataset_name} dataset")
    return num_clusters, dataset, dataset_name, with_numba


def check_required(conf_type, conf_enabled, tag):
    return (conf_type == 'required' or tag != 'required_only') and conf_enabled


def run_serial(runner: KMeansRunner, kmeans_type: str, config_key: List[Dict], tag: str) -> None:
    """ Runs the KMeans serial version for the specified configuration. """
    # Initialize
    for conf in config_key:
        if check_required(conf['type'], conf['enabled'], tag):
            # Extract the properties
            num_clusters, dataset, dataset_name, with_numba = prepare_for_run(kmeans_type, conf)
            # Run K-Means
            if with_numba:
                kmeans_type += '_numba'
                run_with_numba(kmeans_obj=runner, run_type=kmeans_type, num_clusters=num_clusters,
                               dataset=dataset)
            else:
                runner.run(run_type=kmeans_type, num_clusters=num_clusters, dataset=dataset)


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
    kmeans_runner = KMeansRunner()
    for config_key in conf.config_keys:
        run_serial(runner=kmeans_runner, kmeans_type=config_key,
                   config_key=conf.get_config(config_name=config_key), tag=conf.tag)

    main_logger.info("Assignment 3 Finished")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
