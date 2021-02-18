#!/bin/bash

# Get the absolute real path of the script and cd one directory up
# This is necessary for working even from inside symlinks
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH/..
# This the default conda env that `make install` creates
CONDA_BIN="~/anaconda3/envs/dse512_playground/bin/"
# Construct the command and run it with `eval`
command="${CONDA_BIN}python assignment1/assignment1.py -c confs/assignment1.yml -l logs/assignment1.log"
eval $command