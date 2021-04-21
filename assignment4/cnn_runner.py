import os
from typing import Dict, IO, Union

import torch
import torchvision
from torch.utils.data import DataLoader

from playground import ColorizedLogger, profileit, timeit


class CnnRunner:
    logger: ColorizedLogger
    outputs_file: IO
    dataset: Dict
    epochs: int
    batch_size: int
    learning_rate: float
    momentum: float

    def __init__(self, dataset: Dict, epochs: int, batch_size: int, learning_rate: float,
                 momentum: float = 0, seed: int = 1):
        self.logger = ColorizedLogger(f'CnnRunner', 'green')
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        torch.backends.cudnn.enabled = False
        torch.manual_seed(seed)

    def dataset_loader(self, num_processes: int):
        if self.dataset['name'].lower() == 'mnist':
            transformation = torchvision.transforms\
                                        .Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                                  ])
            mnist_dataset = torchvision.datasets.MNIST(self.dataset['save_path'],
                                                       train=True,
                                                       download=True,
                                                       transform=transformation)
            train_loader = torch.utils.data.DataLoader(mnist_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       num_workers=num_processes)
        else:
            raise Exception("Dataset not yet supported!")

        return train_loader

    def run(self, num_processes: int):
        """

        Args:
            num_processes:

        Returns:

        """

        train_loader = self.dataset_loader(num_processes)
        return
        # Prepare output folders and names
        # sys_path = os.path.dirname(os.path.realpath(__file__))
        # output_file_name = f'assignment4_{dataset}_{num_processes}.txt'
        # profiler_file_name = f'assignment4_{dataset}_{num_processes}.o'
        # output_base_path = os.path.join(sys_path, '..', 'outputs')
        # if not os.path.exists(output_base_path):
        #     os.makedirs(output_base_path)
        # profiler_file_path = os.path.join(output_base_path, profiler_file_name)
        # output_file_path = os.path.join(output_base_path, output_file_name)
        #
        # # Open results output file
        # with open(output_file_path, 'w') as self.outputs_file:
        #     self.outputs_file.write(f'CNN for the {dataset} dataset '
        #                             f'with {num_processes} num_processes .\n')
        #
        #     # Load Dataset if not already loaded
        #     # features = self._load_dataset(dataset)
        #
        #     # Run Kmeans
        #     k_words = ['kmeans.py', 'ncalls']  # Include only pstats that contain these words
        #     custom_print = f'Profiling CNN for the `{dataset}` dataset' + \
        #                    f'with {num_processes} num_processes: '
        #     with profileit(file=self.outputs_file, profiler_output=profiler_file_path,
        #                    custom_print=custom_print,
        #                    keep_only_these=k_words):
        #         # Save results
        #         self.logger.info(f"Results: \n{results}")
        #         pass
