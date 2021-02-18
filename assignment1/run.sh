#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH/..
ls
CONDA_BIN="~/anaconda3/envs/dse512_playground/bin/"
command="${CONDA_BIN}python assignment1/assignment1.py -c confs/assignment1.yml -l logs/assignment1.log"
eval $command