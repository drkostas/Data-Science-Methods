#!/bin/bash
which python

python parallel_bench.py -t all -n 1 -n 2 -n 4 -n 8 -n 16 -m 345.678 -m 123.456 -l 1000000 -l 1500000