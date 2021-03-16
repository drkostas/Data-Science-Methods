#!/bin/bash

real_path="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P )"

qsub ${real_path}/kmeans_vectorized_iris_nodes1ppn1.pbs
qsub ${real_path}/kmeans_distributed_iris_nodes2ppn10.pbs
qsub ${real_path}/kmeans_vectorized_tcga_nodes1ppn1.pbs
qsub ${real_path}/kmeans_distributed_tcga_nodes2ppn10.pbs
