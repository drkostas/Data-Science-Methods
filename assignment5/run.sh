#!/bin/bash

# --- Configure Paths --- #
# Conda Paths
CONDA_SRC="/nics/b/home/kgeorgio/.conda"
CONDA_DIST="/conda"
CONDA_BIN_DIST="${CONDA_DIST}/envs/dse512_playground/bin/"
# Project Paths
PROJ_PATH_SRC="/lustre/haven/proj/UTK0150/kgeorgio/playground"
PROJ_PATH_DIST="/playground"
# Executable, conf, log file paths
KMEANS_VECT_EXECUTABLE_DIST="${PROJ_PATH_DIST}/assignment2/assignment2.py"
KMEANS_VECT_CONFIG_DIST="${PROJ_PATH_DIST}/confs/assignment5_kmeans_vectorized.yml"
KMEANS_VECT_LOG_DIST="${PROJ_PATH_DIST}/logs/assignment5_kmeans_vectorized.log"

# --- Construct the command with kmeans_vectorized --- #
KMEANS_VECT_COMMAND="${CONDA_BIN_DIST}python ${KMEANS_VECT_EXECUTABLE_DIST} -c ${KMEANS_VECT_CONFIG_DIST} -l ${KMEANS_VECT_LOG_DIST}"

# --- Run it in the container
singularity exec --bind "$CONDA_SRC":"$CONDA_DIST" --bind "$PROJ_PATH_SRC":"$PROJ_PATH_DIST" pydatacpu.sif "$KMEANS_VECT_COMMAND"
