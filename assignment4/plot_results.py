import os
import logging
import traceback
from glob import glob
import numpy as np

from playground import ColorizedLogger

main_logger = ColorizedLogger('Plotter', 'yellow')


def main():
    sys_path = os.path.dirname(os.path.realpath(__file__))
    output_base_path = os.path.join(sys_path, '..', 'outputs', 'assignment4')
    # Find max run number and set the next
    all_runs = [d for d in glob(os.path.join(output_base_path, "run*"))
                if os.path.isdir(d)]
    if len(all_runs) > 0:
        all_runs = [d.split(os.sep)[-1] for d in all_runs]
        run_folder_name = max(all_runs)
    else:
        raise Exception("No runs found!")
    # Create outputs folder for this run
    run_specific_path = os.path.join(output_base_path, run_folder_name, '1_Processes')

    print(np.load(file=os.path.join(run_specific_path, "metadata.npy"), allow_pickle=True))
    print(np.load(file=os.path.join(run_specific_path, "train_epoch_accuracies.npy"), allow_pickle=True))
    print(np.load(file=os.path.join(run_specific_path, "train_epoch_losses.npy"), allow_pickle=True))
    print(np.load(file=os.path.join(run_specific_path, "train_iter_losses.npy"), allow_pickle=True))
    print(np.load(file=os.path.join(run_specific_path, "train_epoch_times.npy"), allow_pickle=True))
    # print(np.load(file=os.path.join(run_specific_path, "test_results_before.npy"), allow_pickle=True))
    print(np.load(file=os.path.join(run_specific_path, "test_results_after.npy"), allow_pickle=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
