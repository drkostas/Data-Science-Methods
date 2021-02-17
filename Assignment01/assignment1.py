import logging
import traceback

from playground.main import _setup_log, _argparser
from playground.fancy_log.colorized_log import ColorizedLog
from playground.configuration.configuration import Configuration


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


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
