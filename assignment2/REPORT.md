# Assignment02 - K-Means Report 
Report Location: https://github.com/drkostas/DSE512-playground/blob/master/assignment2/REPORT.md

## TL;DR
- The code is in `/lustre/haven/proj/UTK0150/kgeorgio/playground/assignment2` and 
at `https://github.com/drkostas/DSE512-playground/tree/master/assignment2` in Github.
- Detailed logs at `/lustre/haven/proj/UTK0150/kgeorgio/playground/logs/final` and their simplified 
jists at `/lustre/haven/proj/UTK0150/kgeorgio/playground/outputs/final` (they're in Github too).
- The different implementations gave the same results and their correspoind times are shown in the 
table below:
  
| Dataset | Jacob's Vectorized | My Vectorized  | Distributed(20procs) |
|---------|--------------------|----------------|----------------------|
| Iris    | 0.05377 sec(s)     | 0.00288 sec(s) | 0.01159 sec(s)       |
| TCGA    | 15.92928 sec(s)    | 6.26668 sec(s) | 3.09491 sec(s)       |

## Detailed Report
### Paths
The repository that hosts both of the assignments, along with the code I wrote while experimenting with 
the stuff we talked about in the lectures can be found in isaac 
(`/lustre/haven/proj/UTK0150/kgeorgio/playground`) and 
[github](https://github.com/drkostas/DSE512-playground). More specifically the code for assignment2 
and the `pbs scripts` can be found in `playground/assignment2`, the configuration files in 
`playground/confs`, the log files in `playground/logs/final`, and the sanitized summaries of each job 
in `playground/outputs/final`.

### Setting Up
The code needed to set up the environment and install the requirements is in the `playground/Makefile`.

I just run `make install server=prod`

### Code Contents

#### assignment2.py
The main python file used to call each K-Means version (`vectorized`, `distributed` e.t.c.) is 
`playground/assignment2/assignment2.py` and is basically a wrapper. This file takes two basic arguments 
(`-c`: path to config, `-l`: where to write the logs). Depending on the configuration file, it 
calls `playground/assignment2/kmeans.py` using either the `mpirun` command or the `python` command. 

### kmeans.py
The actual K-Means code is in the `playground/assignment2/kmeans.py` which contains the `KMeansRunner`
class that has all the different K-Means implementations. This file is called by 
`playground/assignment2/assignment2.py` using the following arguments:
- `-k`: Number of clusters
- `-d`: Dataset to use
- `-t`: Type of kmeans to run

The available K-Means implementation that correspond to different functions in this file are:
- `simple`: Jacob's in-class simple implementation 
- `vectorized`: My vectorized implementation
- `vectorized_jacob`: Jacob's in-class vectorized implementation
- `distributed`: My distributed implementation
- `distributed_jacob`: Jacob's in-class distributed implementation

#### Configurations
The final configurations used for this assignment are all located in `playground/confs` and are the 
following:
- `assignment2_isaac_iris_nodes1ppn1.yml`: Runs my and Jacob's vectorized versions 
for the `Iris` dataset (just `python` - no `mpirun`)
- `assignment2_isaac_tcga_nodes1ppn1.yml`: Runs my and Jacob's vectorized versions 
for the `TCGA` dataset (just `python` - no `mpirun`)
- `assignment2_isaac_iris_nodes2ppn10.yml`: Runs my vectorized version for the 
  `Iris` dataset (with 20 `nprocs` - using `mpirun`)
- `assignment2_isaac_tcga_nodes2ppn10.yml`: Runs my vectorized version for the 
  `TCGA` dataset (with 20 `nprocs` - using `mpirun`)
  
#### PBS Scripts
The PBS scripts used to submit the corresponding jobs are all located in the `playground/assignment2` 
folder and are the following:
- `kmeans_vectorized_iris_nodes1ppn1.pbs`: Calls `assignment2.py` using 
  `assignment2_isaac_iris_nodes1ppn1.yml` for `1 node - 1 ppn`
- `kmeans_vectorized_tcga_nodes1ppn1.pbs`: Calls `assignment2.py` using 
  `assignment2_isaac_tcga_nodes1ppn1.yml` for `1 node - 1 ppn`
- `kmeans_vectorized_iris_nodes2ppn10.pbs`: Calls `assignment2.py` using 
  `assignment2_isaac_iris_nodes2ppn10.yml` for `1 node - 20 ppn(s)`
- `kmeans_vectorized_tcga_nodes2ppn10.pbs`: Calls `assignment2.py` using 
  `assignment2_isaac_tcga_nodes2ppn10.yml` for `1 node - 20 ppn(s)`
  
The last two PBS scripts where supposed to run for `2 nodes - 10 ppn(s)` but I was getting some 
strange errors even though I fixed the original error.
To call all the PBS scripts at once I run: `playground/assignment2/run.sh`

### Results
The full `.o` and `.e` outputs of the above jobs are stored in `playground/logs/final`. They include 
ANSI colors, so they should be opened using either `cat` or `less -R`.

I highly recommend you check the simplified results which include the run time, and the final 
assignments and centroids - refer to `playground/outputs/final`. There, you will find 6 `txt` files 
with the results of each of the 6 runs (3 implementations x 2 datasets). The naming of these `txt` 
files are identical to the configuration files.

**The `Total Time` refers to the time K-Means run after loading the dataset.**

#### Findings

After reviewing the output txt files, you can see that the `assignments` are completely identical. 
The `centroids` have small differences due to different floating-point precisions used in the 
distributed version, but it doesn't really matter as the assignments are the same.

#### Times

| Dataset | Jacob's Vectorized | My Vectorized  | Distributed(20procs) |
|---------|--------------------|----------------|----------------------|
| Iris    | 0.05377 sec(s)     | 0.00288 sec(s) | 0.01159 sec(s)       |
| TCGA    | 15.92928 sec(s)    | 6.26668 sec(s) | 3.09491 sec(s)       |

As you can see in tha table above, for the Iris dataset, the distributed version is ~5 times faster 
than the in-class vectorized implementation, but slower than the optimized vectorized version. This 
was probably due to `MPI`'s overhead that didn't give enough speedup for the relatively small `Iris` 
dataset.
In contrast, for the TCGA dataset, which is significantly larger, we can see that the distributed 
version was again ~5 times faster than the simple vectorized version and ~2 times faster the optimized 
vectorized version.