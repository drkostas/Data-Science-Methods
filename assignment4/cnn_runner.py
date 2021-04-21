import os
from typing import Dict, IO, Union
from tqdm import tqdm
# Torch & Torch Vision
import torch
from torch import nn, optim, backends
from torch.nn import functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Sampler
# Distributed Torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import DistributedOptimizer
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


class CnnRunner:
    logger: ColorizedLogger
    outputs_file: IO
    dataset: Dict
    epochs: int
    batch_size: int

    def __init__(self, dataset: Dict, epochs: int, batch_size: int, learning_rate: float,
                 momentum: float = 0, seed: int = 1):
        # Set the object variables
        self.logger = ColorizedLogger(f'CnnRunner', 'green')
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        # Configure torch variables
        backends.cudnn.enabled = False
        torch.manual_seed(seed)
        # Create the training modules
        self.my_model = LeNet5(num_classes=10)
        self.logger.info(f"Model Architecture:\n{self.my_model}")
        self.optimizer = optim.SGD(self.my_model.parameters(),
                                   lr=learning_rate, momentum=momentum)
        self.loss_function = nn.CrossEntropyLoss()
        self.logger.info("Model parameters are configured.")

    def dataset_loader(self):
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
                                        download=True)
        else:
            raise Exception("Dataset not yet supported!")
        self.logger.info(f"{self.dataset['name'].capitalize()} dataset loaded successfully.")

        return mnist_train, mnist_test

    def print_train_results(self, epoch_losses, iter_losses, epoch_times):
        self.logger.info(f"epoch_losses (len {len(epoch_losses)}):\n{epoch_losses}")
        self.logger.info(f"iter_losses (len {len(iter_losses)}):\n{iter_losses}")
        self.logger.info(f"epoch_times (len {len(epoch_times)}):"
                         f"\n{epoch_times}")

    def print_test_results(self, test_loss, correct, total, percent_correct):
        self.logger.info(f"test_loss: {test_loss}")
        self.logger.info(f"correct/total : {correct}/{total}")
        self.logger.info(f"percent_correct: {percent_correct:.2f}%")

    def train_non_parallel(self, train_loader):
        # TODO: Reset model grads because Im using same instance for training on different N processors
        self.my_model.train()
        iter_losses = []
        epoch_losses = []
        epoch_times = []

        iter_epochs = tqdm(range(self.epochs), desc='Training Epochs')
        for _ in iter_epochs:
            timeit_ = timeit(internal_only=True)
            epoch_loss = 0.0
            num_mini_batches = 0
            with timeit_:
                # iter_mini_batches = tqdm(enumerate(train_loader), desc='Mini Batches', leave=False)
                iter_mini_batches = enumerate(train_loader)
                for num_mini_batches, (X, Y) in iter_mini_batches:
                    self.optimizer.zero_grad()
                    pred = self.my_model(X)
                    loss = self.loss_function(pred, Y)
                    iter_loss = loss.item()
                    iter_losses.append(iter_loss)
                    # iter_mini_batches.set_postfix(iter_loss=iter_loss)
                    epoch_loss += iter_loss
                    loss.backward()
                    self.optimizer.step()

            epoch_loss /= (num_mini_batches + 1)
            epoch_losses.append(epoch_loss)
            epoch_time = timeit_.total
            epoch_times.append(epoch_time)
            iter_epochs.set_postfix(epoch_loss=epoch_loss, epoch_time=epoch_time)

        return epoch_losses, iter_losses, epoch_times

    def test_non_parallel(self, test_loader):
        self.my_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            iter_mini_batches = tqdm(enumerate(test_loader), desc='Testing', leave=False)
            for num_mini_batches, (X, Y) in iter_mini_batches:
                pred_output = self.my_model(X)
                test_loss += self.loss_function(pred_output, Y)
                pred = pred_output.data.max(1, keepdim=True)[1]
                correct += pred.eq(Y.data.view_as(pred)).sum()
                iter_mini_batches.set_postfix(test_loss_accum=test_loss)
        test_loss /= len(test_loader.dataset)
        total = len(test_loader.dataset)
        percent_correct = 100. * correct / len(test_loader.dataset)

        return test_loss, correct, total, percent_correct

    def run_non_parallel(self, mnist_train, num_processes: int):
        self.logger.info("Non-parallel mode requested..")

        train_loader = torch.utils.data.DataLoader(mnist_train,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=num_processes)
        # Test with randomly initialize parameters
        test_loss, correct, total, percent_correct = self.test_non_parallel(train_loader)
        self.logger.info("Randomly Initialized params testing:")
        self.print_test_results(test_loss, correct, total, percent_correct)

        # Training
        epoch_losses, iter_losses, epoch_times = self.train_non_parallel(train_loader)
        self.logger.info("Training Finished! Results:")
        self.print_train_results(epoch_losses, iter_losses, epoch_times)

        # Testing
        test_loss, correct, total, percent_correct = self.test_non_parallel(train_loader)
        self.logger.info("Testing Finished! Results:")
        self.print_test_results(test_loss, correct, total, percent_correct)

        return epoch_losses, iter_losses, epoch_times, test_loss, correct, total, percent_correct

    def run_data_parallel(self, mnist_train, num_processes: int):
        self.logger.info("Data parallel mode requested..")
        sampler = DistributedSampler(mnist_train)
        train_loader = DataLoader(mnist_train,
                                  batch_size=self.batch_size,
                                  sampler=sampler,
                                  num_workers=num_processes)
        my_parallel_model = DistributedDataParallel(self.my_model)
        # TODO: Create the training loop

    def run(self, num_processes: int, data_parallel: bool):
        """

        Args:
            num_processes:
            data_parallel:
        Returns:

        """

        mnist_train, mnist_test = self.dataset_loader()
        if data_parallel:
            self.run_data_parallel(mnist_train, num_processes)
        else:
            epoch_losses, iter_losses, epoch_times, test_loss, correct, total, percent_correct = \
                self.run_non_parallel(mnist_train, num_processes)

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
