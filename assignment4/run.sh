#!/bin/bash

real_path="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P )"

qsub ${real_path}/mnist_with_lenet5_non_parallel_1.pbs
qsub ${real_path}/mnist_with_lenet5_parallel_1.pbs
qsub ${real_path}/mnist_with_lenet5_parallel_2.pbs
qsub ${real_path}/mnist_with_lenet5_parallel_4.pbs
qsub ${real_path}/mnist_with_lenet5_parallel_8.pbs

