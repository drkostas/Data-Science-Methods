#!/bin/bash

# --- Configure Paths --- #
# Conda Paths
CONDA_SRC="/nics/b/home/kgeorgio/.conda"
CONDA_DEST="/conda"
CONDA_BIN_DEST="${CONDA_DEST}/envs/dse512_playground/bin/"
# Project Paths
PROJ_PATH_SRC="/lustre/haven/proj/UTK0150/kgeorgio/playground"
PROJ_PATH_DEST="/playground"
# Executable, conf, log file paths
KMEANS_VECT_EXECUTABLE_DEST="${PROJ_PATH_DEST}/assignment2/assignment2.py"
KMEANS_VECT_CONFIG_DEST="${PROJ_PATH_DEST}/confs/assignment5_kmeans_vectorized.yml"
KMEANS_VECT_LOG_DEST="${PROJ_PATH_DEST}/logs/assignment5_kmeans_vectorized.log"

# --- Construct the command with kmeans_vectorized --- #
KMEANS_VECT_COMMAND="${CONDA_BIN_DEST}python ${KMEANS_VECT_EXECUTABLE_DEST} -c ${KMEANS_VECT_CONFIG_DEST} -l ${KMEANS_VECT_LOG_DEST}"

# --- Run it in the container
SINGULARITY_COMMAND="singularity exec --bind ${CONDA_SRC}:${CONDA_DEST} --bind ${PROJ_PATH_SRC}:${PROJ_PATH_DEST} pydatacpu.sif ${KMEANS_VECT_COMMAND}"
eval $SINGULARITY_COMMAND