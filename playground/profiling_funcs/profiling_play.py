from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from playground import ColorizedLogger, timeit


class ProfilingPlay:
    logger: ColorizedLogger

    def __init__(self, log_name: str):
        ColorizedLogger.setup_logger(log_path=log_name, clear_log=False)
        self.logger = ColorizedLogger(f'ProfilePlay', 'blue')
        self.logger.info(f"Initialized ProfilingPlay..")

    def hello(self):
        self.logger.info("Hello World!")

    @staticmethod
    def load_boston_data():
        boston = datasets.load_boston()
        return boston.data, boston.target

    @staticmethod
    def build_model():
        hparams = {
            'n_estimators': 500,
            'max_depth': 5,
            'min_samples_split': 5,
            'learning_rate': 0.01,
            'loss': 'ls'
        }

        model = GradientBoostingRegressor(**hparams)

        return model

    @classmethod
    def load_data(cls):
        data, target = cls.load_boston_data()

        x_train, x_valid, y_train, y_valid = train_test_split(
            data, target, test_size=0.33, random_state=42
        )

        return x_train, x_valid, y_train, y_valid

    def boston_gbpm(self):
        x_train, x_valid, y_train, y_valid = self.load_data()

        model = self.build_model()

        model.fit(x_train, y_train)
        preds = model.predict(x_valid)

        mse = mean_squared_error(y_valid, preds)
        self.logger.info(f"The mean squared error (MSE) on test set: {mse:.4f}")

    @staticmethod
    def build_list():
        return [x for x in range(1_000_000)]

    @staticmethod
    def exponentiate(arry, power):
        return [x ** power for x in arry]

    def run_exponentiate(self):
        my_list = self.build_list()
        squared = self.exponentiate(my_list, 2)

    @timeit(custom_print="Running run() for {args[1]!r} took {duration:2.5f} sec(s)")
    def run(self, func_to_run: str):
        self.logger.info(f"Starting function {func_to_run}")
        if func_to_run == 'hello_world':
            self.hello()
        elif func_to_run == 'boston_gbpm':
            self.boston_gbpm()
        elif func_to_run == 'exponentiate':
            self.run_exponentiate()
        else:
            raise NotImplementedError(f"Function {func_to_run} not yet implemented.")


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, required=True, help='Which function to run',
                        choices=['hello_world', 'boston_gbpm', 'exponentiate'])
    parser.add_argument('-p', type=str, required=True, help='Profiling Type',
                        choices=['internal', 'external', 'disabled'])
    parser.add_argument('-s', type=str, required=True, help='Sort stats by',
                        choices=['tottime', 'cumtime'])
    parser.add_argument('-l', type=str, required=False, default='profiling.log', help='Log File Name')
    args = parser.parse_args()

    return args.f, args.p, args.l, args.s


if __name__ == "__main__":
    import argparse

    # Read and Parse arguments
    function_to_run, profiling_type, log_name, sort_stats = setup_args()
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
        stats = pstats.Stats(profiler).sort_stats(sort_stats)
        stats.print_stats()
