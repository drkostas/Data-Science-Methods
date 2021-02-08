# DSE-512 Playground

## Table of Contents

+ [About](#about)
+ [Getting Started](#getting_started)
    + [Prerequisites](#prerequisites)
+ [Running locally](#run_locally)
    + [Execution Options](#execution_options)

## About <a name = "about"></a>

A playground repo for the DSE-512 course.

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing
purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites <a name = "prerequisites"></a>

You need to have a machine with Python > 3.6 and any Bash based shell (e.g. zsh) installed.

```
$ python3.6 -V
Python 3.6.9

echo $SHELL
/usr/bin/zsh
```

Then create a virtual environment using either virtualenv or anaconda:

<b>Virtualenv</b>
```bash
$ python3 -m venv dse512_playground
``` 

<b>Conda</b>
```bash
$ conda create --name dse512_playground
Collecting package metadata (current_repodata.json): done
Solving environment: done
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
``` 

## Running the code locally <a name = "run_locally"></a>

First, make sure you are in the created virtual environment:

<b>For Virtualenv</b>
```bash
$ source dse512_playground/bin/activate
(dse512_playground) 
~/drkostas/Projects/DSE512-playground 

$ which python
~/drkostas/Projects/DSE512-playground/dse512_playground/bin/python
(dse512_playground) 
``` 

<b>For Conda</b>
```bash
$ conda create --name dse512_playground
Collecting package metadata (current_repodata.json): done
Solving environment: done
Preparing transaction: done
Verifying transaction: done
Executing transaction: done

$ conda activate DSE512-playground
(DSE512-playground)

$ which python
/home/drkostas/anaconda3/envs/DSE512-playground/bin/python
(DSE512-playground)
``` 

### Execution Options <a name = "execution_options"></a>

Depending on the file you want to run, you'll need to follow the corresponding instructions. To view them, just run:

```bash
$ python <your file name>.py --help
usage: <your file name>.py -m {run_mode_1,run_mode_2,run_mode_3} -c CONFIG_FILE [-l LOG]
               [-d] [-h]

<Your python file\'s description.

required arguments:
  -m {run_mode_1,run_mode_2,run_mode_3}, --run-mode {run_mode_1,run_mode_2,run_mode_3}
                        Description of the run modes
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        The configuration yml file
  -l LOG, --log LOG     Name of the output log file

optional arguments:
  -d, --debug           enables the debug log messages
```

To run it following the instructions.