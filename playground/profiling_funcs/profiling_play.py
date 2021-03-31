# import sys
# import os
# from typing import Dict
# import numpy as np

from playground import ColorizedLogger


class ProfilingPlay:
    logger: ColorizedLogger

    def __init__(self, log_name: str):
        ColorizedLogger.setup_logger(log_path=log_name, clear_log=False)
        self.logger = ColorizedLogger(f'ProfilePlay', 'blue')
        self.logger.info(f"Initialized ProfilingPlay..")

    def hello(self):
        self.logger.info("Hello World!")

    def run(self, func_to_run: str):
        self.logger.info(f"Starting function {func_to_run}")
        if func_to_run == 'hello_world':
            self.hello()


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, required=True, help='Which function to run',
                        choices=['hello_world', 'boston_gbpm', 'exponentiate'])
    parser.add_argument('-p', type=str, required=True, help='Profiling Type',
                        choices=['internal', 'external', 'disabled'])
    parser.add_argument('-l', type=str, required=False, default='profiling.log', help='Log File Name')
    args = parser.parse_args()

    return args.f, args.p, args.l


if __name__ == "__main__":
    import argparse

    # Read and Parse arguments
    function_to_run, profiling_type, log_name = setup_args()
    if profiling_type == 'internal':
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()

    # Initialize and run ProfilingPlay
    prof_play = ProfilingPlay(log_name=log_name)
    prof_play.run(func_to_run=function_to_run)

    if profiling_type == 'internal':
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')
        stats.print_stats()
