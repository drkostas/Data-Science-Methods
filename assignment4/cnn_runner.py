import os
import sys
import psutil
from typing import List, Dict, IO, Tuple, Union
import traceback
import argparse
import logging
from glob import glob
from tqdm import tqdm
import numpy as np
from pympler import muppy, summary
# Torch & Torch Vision
import torch
from torch import nn, optim, backends
from torch.nn import functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
# Custom stuff
from playground import ColorizedLogger, timeit


class LeNet5(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.conv_2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.conv_3 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.linear_1 = nn.Linear(120, 84)
        self.linear_2 = nn.Linear(84, num_classes)

    def forward(self, input_):
        cnn_step = F.avg_pool2d(F.relu(self.conv_1(input_)),
                                kernel_size=(2, 2), stride=2)
        cnn_step = F.avg_pool2d(F.relu(self.conv_2(cnn_step)),
                                kernel_size=(2, 2), stride=2)
        cnn_step = F.relu(self.conv_3(cnn_step))

        linear_step = torch.flatten(cnn_step, 1)
        linear_step = F.relu(self.linear_1(linear_step))
        linear_step = self.linear_2(linear_step)

        return F.log_softmax(linear_step, dim=-1)


class VolModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 4, 3)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 10, 3)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x


class CnnRunner:
    logger: ColorizedLogger
    outputs_file: IO
    dataset: Dict
    epochs: int
    learning_rate: float
    momentum: float
    batch_size_train: int
    batch_size_test: int
    test_before_train: bool
    results_path: str
    data_parallel: bool
    rank: Union[int, None]

    def __init__(self, dataset: Dict, epochs: int, batch_size_train: int, batch_size_test: int,
                 learning_rate: float, test_before_train: bool, momentum: float = 0,
                 seed: int = 1, data_parallel: bool = False, log_path: str = None):
        # Set the object variables
        self.data_parallel = data_parallel
        if self.data_parallel:
            self.rank = dist.get_rank()
        else:
            self.rank = None
        if self.rank in (None, 0):
            if log_path:
                self.__log_setup(log_path=log_path, clear_log=True)
            self.logger = ColorizedLogger(f'CnnRunner', 'green')
        self.dataset = dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.test_before_train = test_before_train

        # Configure torch variables
        backends.cudnn.enabled = False
        torch.manual_seed(seed)
        # Create the training modules
        self.my_model = LeNet5(num_classes=10)
        # self.my_model = VolModel(num_classes=10)

        self.loss_function = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.rank in (None, 0):
            self.logger.info("Model parameters are configured.")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Model Architecture:\n{self.my_model}")
        # Create folder where the results are going to be saved
        if self.rank in (None, 0):
            self.results_path = self.create_results_folder()

    @staticmethod
    def __log_setup(log_path: str, clear_log: bool = False):
        sys_path = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(sys_path, '..', 'logs', log_path)
        ColorizedLogger.setup_logger(log_path=log_path, clear_log=clear_log)

    @staticmethod
    def create_results_folder():
        # Create Base Assignment folder
        sys_path = os.path.dirname(os.path.realpath(__file__))
        output_base_path = os.path.join(sys_path, '..', 'outputs', 'assignment4')
        # Find max run number and set the next
        previous_runs = [d for d in glob(os.path.join(output_base_path, "run*"))
                         if os.path.isdir(d)]
        if len(previous_runs) > 0:
            previous_runs = [int(d.split(os.sep)[-1][3:]) for d in previous_runs]
            max_run_num = max(previous_runs) + 1
        else:
            max_run_num = 0
        # Create outputs folder for this run
        run_folder_name = f"run{max_run_num}"
        run_specific_path = os.path.join(output_base_path, run_folder_name)
        if not os.path.exists(run_specific_path):
            os.makedirs(run_specific_path)
        return run_specific_path

    def store_results(self, data: Union[Tuple, Dict], num_processes: int, train: bool) -> None:
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        # Create Run Specific Metadata file
        metadata = {"num_processes": num_processes,
                    "epochs": self.epochs,
                    "learning_rate": self.learning_rate,
                    "momentum": self.momentum,
                    "batch_size_train": self.batch_size_train,
                    "batch_size_test": self.batch_size_test,
                    "data_parallel": self.data_parallel}
        # Save metadata as numpy dict and as human-readable csv
        np.save(file=os.path.join(self.results_path, "metadata.npy"), arr=np.array(metadata))
        metadata_csv = np.array([tuple(metadata.keys()), tuple(metadata.values())], dtype=str)
        np.savetxt(os.path.join(self.results_path, "metadata.csv"), metadata_csv,
                   fmt="%s", delimiter=",")

        if train:
            np.save(file=os.path.join(self.results_path, "train_epoch_accuracies.npy"),
                    arr=np.array(data[0]))
            np.save(file=os.path.join(self.results_path, "train_epoch_losses.npy"),
                    arr=np.array(data[1]))
            np.save(file=os.path.join(self.results_path, "train_epoch_times.npy"),
                    arr=np.array(data[2]))
        else:
            for conf_key in data:
                subset = data[conf_key]
                dict_to_save = {"test_loss": subset[0],
                                "correct": subset[1],
                                "total": subset[2],
                                "percent_correct": subset[3]}
                np.save(file=os.path.join(self.results_path, f"test_results_{conf_key}.npy"),
                        arr=np.array(dict_to_save))

    def dataset_loader(self) -> Tuple[datasets.MNIST, datasets.MNIST]:
        if self.dataset['name'].lower() == 'mnist':
            transformation = transforms \
                .Compose([transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ])
            mnist_train = datasets.MNIST(self.dataset['save_path'],
                                         train=True,
                                         download=True,
                                         transform=transformation)
            mnist_test = datasets.MNIST(self.dataset['save_path'],
                                        train=False,
                                        download=True,
                                        transform=transformation)
        else:
            raise NotImplemented("Dataset not yet supported!")
        if self.rank in (None, 0):
            self.logger.info(f"{self.dataset['name'].capitalize()} dataset loaded successfully.")

        return mnist_train, mnist_test

    def print_test_results(self, test_loss: float, correct: int, total: int,
                           percent_correct: float) -> None:
        self.logger.info(f"Test Loss: {test_loss}", color="blue")
        self.logger.info(f"Correct/Total : {correct}/{total}", color="blue")
        self.logger.info(f"Accuracy: {100 * percent_correct:.2f}%", color="blue")

    def train_non_parallel(self, train_loader: DataLoader) -> Tuple[List, List, List]:
        size_train_dataset = len(train_loader.dataset)
        epoch_losses = []
        epoch_accuracies = []
        epoch_times = []
        optimizer = optim.SGD(self.my_model.parameters(),
                              lr=self.learning_rate, momentum=self.momentum)
        self.my_model.train()
        iter_epochs = tqdm(range(self.epochs), desc='Training Epochs')
        for _ in iter_epochs:
            timeit_ = timeit(internal_only=True)
            epoch_loss = 0.0
            correct = 0
            num_mini_batches = 0
            with timeit_:
                iter_mini_batches = enumerate(train_loader)
                for num_mini_batches, (X, Y) in iter_mini_batches:
                    optimizer.zero_grad()
                    pred = self.my_model(X)
                    pred_val = torch.flatten(pred.data.max(1, keepdim=True)[1])
                    # correct += pred_val.eq(Y.data.view_as(pred_val)).sum().item()
                    correct += (pred_val == Y).sum().item()
                    loss = self.loss_function(pred, Y)
                    iter_loss = loss.item()
                    epoch_loss += iter_loss
                    loss.backward()
                    optimizer.step()

            epoch_loss /= (num_mini_batches + 1)
            epoch_losses.append(epoch_loss)
            epoch_accuracy = correct / size_train_dataset
            epoch_accuracies.append(epoch_accuracy)
            epoch_time = timeit_.total
            epoch_times.append(epoch_time)
            iter_epochs.set_postfix(epoch_accuracy=epoch_accuracy, epoch_loss=epoch_loss,
                                    epoch_time=epoch_time)

        return epoch_accuracies, epoch_losses, epoch_times

    def train_data_parallel(self, train_loader: DataLoader) -> Tuple[List, List, List]:

        my_model = nn.parallel.DistributedDataParallel(self.my_model)
        learning_rate = self.learning_rate * dist.get_world_size()
        optimizer = optim.SGD(my_model.parameters(), lr=learning_rate)

        size_train_dataset = len(train_loader.dataset)
        epoch_losses = []
        epoch_accuracies = []
        epoch_times = []

        self.my_model.train()
        if self.rank == 0:
            iter_epochs = tqdm(range(self.epochs), desc='Training Epochs')
        else:
            iter_epochs = range(self.epochs)

        for _ in iter_epochs:
            timeit_ = timeit(internal_only=True)
            epoch_loss = 0.0
            correct = 0
            num_mini_batches = 0
            with timeit_:
                iter_mini_batches = enumerate(train_loader)
                for num_mini_batches, (X, Y) in iter_mini_batches:
                    optimizer.zero_grad()
                    pred = self.my_model(X)
                    pred_val = torch.flatten(pred.data.max(1, keepdim=True)[1])
                    # correct += pred_val.eq(Y.data.view_as(pred_val)).sum().item()
                    correct += (pred_val == Y).sum().item()
                    loss = self.loss_function(pred, Y)
                    iter_loss = loss.item()
                    epoch_loss += iter_loss
                    loss.backward()
                    optimizer.step()

            epoch_loss /= (num_mini_batches + 1)
            epoch_losses.append(epoch_loss)
            epoch_accuracy = correct / (size_train_dataset / dist.get_world_size())
            epoch_accuracies.append(epoch_accuracy)
            epoch_time = timeit_.total
            epoch_times.append(epoch_time)
            if self.rank == 0:
                iter_epochs.set_postfix(epoch_accuracy=epoch_accuracy, epoch_loss=epoch_loss,
                                        epoch_time=epoch_time)

        return epoch_accuracies, epoch_losses, epoch_times

    def test(self, test_loader: DataLoader) -> Tuple[float, int, int, float]:
        self.my_model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            iter_mini_batches = tqdm(enumerate(test_loader), desc='Testing', leave=False)
            for num_mini_batches, (X, Y) in iter_mini_batches:
                pred = self.my_model(X)
                test_loss += self.loss_function(pred, Y).item()
                pred_val = torch.flatten(pred.data.max(1, keepdim=True)[1])
                # correct += pred_val.eq(Y.data.view_as(pred_val)).sum().item()
                correct += (pred_val == Y).sum().item()
                iter_mini_batches.set_postfix(test_loss_accum=test_loss)
        test_loss /= len(test_loader.dataset)
        size_test_dataset = len(test_loader.dataset)
        accuracy = correct / size_test_dataset

        return test_loss, correct, size_test_dataset, accuracy

    def run_non_parallel(self, train_loader: DataLoader, test_loader: DataLoader) \
            -> Tuple[Tuple, Dict]:

        test_results = {}
        # Test with randomly initialize parameters
        if self.test_before_train:
            test_results["before"] = self.test(test_loader)
            self.logger.info("Randomly Initialized params testing:", color="blue")
            self.print_test_results(*test_results["before"])

        # Training
        train_results = self.train_non_parallel(train_loader)
        self.logger.info("Training Finished! Results:", color="magenta")

        # Testing
        test_results["after"] = self.test(test_loader)
        self.logger.info("Testing Finished! Storing results..", color="blue")
        self.print_test_results(*test_results["after"])

        return train_results, test_results

    def run_data_parallel(self, train_loader: DataLoader, test_loader: DataLoader) \
            -> Tuple[Tuple, Dict]:

        if self.rank == 0:
            self.logger.info(f"World size: {dist.get_world_size()}")

        test_results = {}
        # Test with randomly initialize parameters
        if self.test_before_train:
            test_results["before"] = self.test(test_loader)
            if self.rank == 0:
                self.logger.info("Randomly Initialized params testing:", color="blue")
                self.print_test_results(*test_results["before"])

        # Training
        train_results = self.train_data_parallel(train_loader)
        if self.rank == 0:
            self.logger.info("Training Finished! Results:", color="magenta")

        # Testing
        test_results["after"] = self.test(test_loader)
        if self.rank == 0:
            self.logger.info("Testing Finished! Storing results..", color="blue")
            self.print_test_results(*test_results["after"])

        return train_results, test_results

    def run(self, num_processes: int) -> None:
        """
        Args:
            num_processes:
        Returns:
        """

        # Load the Dataset
        mnist_train, mnist_test = self.dataset_loader()
        # Create Train and Test loaders
        if self.data_parallel:
            mode = "Data Parallel"
            train_sampler = DistributedSampler(mnist_train)
            shuffle = False
        else:
            mode = "Non-parallel"
            train_sampler = None
            shuffle = True
        if self.rank in (None, 0):
            self.logger.info(f"{mode} mode with {num_processes} proc(s) requested..")
        train_loader = torch.utils.data.DataLoader(mnist_train,
                                                   batch_size=self.batch_size_train,
                                                   shuffle=shuffle,
                                                   num_workers=num_processes,
                                                   sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(mnist_test,
                                                  batch_size=self.batch_size_test,
                                                  shuffle=True,
                                                  num_workers=num_processes)
        # Train and Test
        if self.data_parallel:
            train_results, test_results = self.run_data_parallel(train_loader, test_loader)
        else:
            train_results, test_results = self.run_non_parallel(train_loader, test_loader)
        # Save Results
        if self.rank in (None, 0):
            self.store_results(data=train_results, num_processes=num_processes, train=True)
            self.store_results(data=test_results, num_processes=num_processes, train=False)


def get_args() -> argparse.Namespace:
    """Setup the argument parser

    Returns:
        argparse.Namespace:
    """
    parser = argparse.ArgumentParser(
        description='A playground repo for the DSE-512 course..',
        add_help=False)
    # Required Args
    required_args = parser.add_argument_group('Required Arguments')
    # Optional args
    required_args.add_argument('--dataset-name')
    required_args.add_argument('--dataset-path')
    required_args.add_argument('-ep', '--epochs')
    required_args.add_argument('-btr', '--batch-size-train')
    required_args.add_argument('-bte', '--batch-size-test')
    required_args.add_argument('-lr', '--learning-rate')
    required_args.add_argument('-mo', '--momentum')
    required_args.add_argument('-l', '--log')
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument('--test-before-train', action='store_true')
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


def main():
    import mpi4py.MPI as MPI
    from os import environ
    # Set default env vars if are not set
    if environ.get('MASTER_ADDR') is None:
        os.environ['MASTER_ADDR'] = 'localhost'
    if environ.get('MASTER_PORT') is None:
        os.environ['MASTER_PORT'] = '12345'
    # Get the arguments
    args = get_args()
    # Setup MPI and Torch distributed
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    dist.init_process_group('gloo',
                            init_method='env://',
                            world_size=world_size,
                            rank=rank)
    # Initialize the CNN Runner
    dataset = {"name": args.dataset_name,
               "save_path": args.dataset_path}
    cr = CnnRunner(dataset=dataset,
                   epochs=int(args.epochs),
                   batch_size_train=int(args.batch_size_train),
                   batch_size_test=int(args.batch_size_test),
                   learning_rate=float(args.learning_rate),
                   momentum=float(args.momentum),
                   test_before_train=args.test_before_train,
                   data_parallel=True,
                   log_path=args.log)
    # Train and Test the model
    cr.run(num_processes=world_size)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
