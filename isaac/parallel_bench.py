import random
import time
import argparse
import multiprocessing as mp
import threading
from concurrent import futures


class BenchTests:
    def __init__(self, max_float, loops):
        random.seed(2)
        self.max_float = max_float
        self.loops = loops
        self.result = random.uniform(1.0, self.max_float)
        self.numbers_to_mult = [random.uniform(1.0, max_float) for _ in range(loops)]
        self.numbers_to_div = [random.uniform(1.0, max_float) for _ in range(loops)]
        self.first_list = self.numbers_to_mult
        self.second_list = self.numbers_to_div
        self.third_list = []

    def math_calc(self):
        while len(self.numbers_to_mult) + len(self.numbers_to_div) > 0:
            try:
                self.result *= self.numbers_to_mult.pop()
            except IndexError as ie:
                pass
            try:
                self.result /= self.numbers_to_div.pop()
            except IndexError as ie:
                pass

    def fill_and_empty_list(self):
        while len(self.first_list) + len(self.second_list) > 0:
            try:
                self.third_list.append(self.first_list.pop())
            except IndexError as ie:
                pass
            try:
                self.third_list.append(self.second_list.pop())
            except IndexError as ie:
                pass


def _argparser():
    parser = argparse.ArgumentParser(
        description='Parallel Processing Benchmarking.',
        add_help=False)
    # Required Args
    required_arguments = parser.add_argument_group('Required Arguments')
    choices = ['math_calc', 'fill_and_empty_list', 'all']
    required_arguments.add_argument('-t', '--tests-to-run', choices=choices,
                                    required=True,
                                    help='Which tests to run.')
    required_arguments.add_argument('-n', '--num-threads-processes', action='append',
                                    help="Number of the threads and processes to use. Can add many.")
    required_arguments.add_argument('-m', '--max-float', action='append',
                                    help="Max float to use on each test. Add one arg for each test.")
    required_arguments.add_argument('-l', '--num-loops', action='append',
                                    help="Number of loops to do on each test. Add one arg for each test.")
    # Optional args
    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument("-h", "--help", action="help",
                          help="When adding -m and -l arguments, add them in the this order: %s" % choices[:-1])

    return parser.parse_args()


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


@timeit
def run_with_multiprocess(func, num_processes, *args):
    processes = []
    for _ in range(num_processes):
        processes.append(
            # Create a process that wraps our function
            mp.Process(target=func, args=(*args,))
        )

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()


@timeit
def run_with_threads(func, num_threads, *args):
    threads = []
    for _ in range(num_threads):
        threads.append(
            # Create a thread that wraps our function
            threading.Thread(target=func, args=(*args,))
        )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


@timeit
def run_with_concurrent(func, num_threads, *args):
    with futures.ThreadPoolExecutor() as executor:
        [executor.submit(func, *args) for _ in range(num_threads)]


def main():
    # Initializing
    args = _argparser()
    test_proc_threads = [int(num) for num in args.num_threads_processes]
    if args.tests_to_run == 'all':
        if len(args.max_float) + len(args.num_loops) < 4:
            raise Exception(
                "You should add the -t and -l options, one time for each test, with the order specified in help.")
    else:
        if len(args.max_float) + len(args.num_loops) < 2:
            raise Exception("You should add the -t and -l options. For more run with -h option.")

    max_float = [float(num) for num in args.max_float]
    num_loops = [int(num) for num in args.num_loops]

    if args.tests_to_run in ['math_calc', 'all']:
        print("\n\nStarting benchmark for `math_calc`..")
        for num in test_proc_threads:
            # Multiprocessing
            math_ops = BenchTests(max_float=max_float[0], loops=num_loops[0])
            print("\nStarting with %s processes:" % num)
            run_with_multiprocess(math_ops.math_calc, num)
            # Threading
            math_ops = BenchTests(max_float=max_float[0], loops=num_loops[0])
            print("Starting with %s threads:" % num)
            run_with_threads(math_ops.math_calc, num)
            # Threading with concurrent
            math_ops = BenchTests(max_float=max_float[0], loops=num_loops[0])
            print("Starting with %s `concurrent` threads:" % num)
            run_with_concurrent(math_ops.math_calc, num)

    if args.tests_to_run in ['fill_and_empty_list', 'all']:
        print("\n\nStarting benchmark for `fill_and_empty_list`..")
        for num in test_proc_threads:
            # Multiprocessing
            math_ops = BenchTests(max_float=max_float[-1], loops=num_loops[-1])
            print("\nStarting with %s processes:" % num)
            run_with_multiprocess(math_ops.fill_and_empty_list, num)
            # Threading
            math_ops = BenchTests(max_float=max_float[-1], loops=num_loops[-1])
            print("Starting with %s threads:" % num)
            run_with_threads(math_ops.fill_and_empty_list, num)
            # Threading with concurrent
            math_ops = BenchTests(max_float=max_float[-1], loops=num_loops[-1])
            print("Starting with %s `concurrent` threads:" % num)
            run_with_concurrent(math_ops.fill_and_empty_list, num)


if __name__ == '__main__':
    main()
