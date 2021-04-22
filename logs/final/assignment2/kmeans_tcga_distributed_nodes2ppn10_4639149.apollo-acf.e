Can't locate local/lib.pm in @INC (you may need to install the local::lib module) (@INC contains: /nics/b/home/kgeorgio/perl5/lib/perl5 /opt/moab/lib/perl5 /opt/moab/lib/perl5 /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/site_perl/5.26.2/x86_64-linux-thread-multi /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/site_perl/5.26.2 /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/5.26.2/x86_64-linux-thread-multi /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/5.26.2 .).
BEGIN failed--compilation aborted.
+ cd /lustre/haven/proj/UTK0150/kgeorgio/assignment2
+ CONDA_BIN=/nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/
+ conf=/lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment2_isaac_tcga_nodes2ppn10.yml
+ log=/lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment2_isaac_tcga_nodes2ppn10.log
+ command='/nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment2.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment2_isaac_tcga_nodes2ppn10.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment2_isaac_tcga_nodes2ppn10.log'
+ eval /nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment2.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment2_isaac_tcga_nodes2ppn10.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment2_isaac_tcga_nodes2ppn10.log
++ /nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment2.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment2_isaac_tcga_nodes2ppn10.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment2_isaac_tcga_nodes2ppn10.log
2021-03-17 22:21:49 Main         INFO     [1m[33mStarting Assignment 2[0m
2021-03-17 22:21:49 Config       INFO     [1m[37mConfiguration file loaded successfully[0m
2021-03-17 22:21:49 Main         INFO     [1m[33mInvoking kmeans.py(distributed) for 10 clusters and the tcga dataset[0m
2021-03-17 22:21:52 KMeans distributed Proc(0) INFO     [1m[34mStarted with 20 processes.[0m
2021-03-17 22:22:05 KMeans distributed Proc(0) INFO     [1m[34mDataset tcga loaded. Shape: (801, 20531).[0m
2021-03-17 22:22:05 Timeit       INFO     [1m[37mInit Time: 0.16858 sec(s)[0m
2021-03-17 22:22:06 Timeit       INFO     [1m[37mFirst Loop Distances Calc Time: 0.13569 sec(s)[0m
2021-03-17 22:22:06 Timeit       INFO     [1m[37mFirst Loop Cluster Assignment Time: 0.00003 sec(s)[0m
2021-03-17 22:22:06 Timeit       INFO     [1m[37mFirst Loop Centroids Update Time: 0.01445 sec(s)[0m
2021-03-17 22:22:08 KMeans distributed Proc(0) INFO     [1m[34mFinal Cluster Assignments: 
[2 6 2 2 7 2 8 2 0 2 1 8 2 9 7 1 6 8 8 2 7 8 6 7 8 6 4 7 0 5 7 7 8 9 2 7 8
 6 9 9 8 2 2 8 8 7 2 4 7 6 7 6 0 2 4 3 7 4 8 1 6 5 1 6 2 4 1 2 8 0 8 1 9 6
 0 6 1 8 2 4 7 2 7 7 2 2 7 7 8 0 2 2 1 9 7 2 4 7 2 1 7 8 1 8 6 8 6 4 6 6 2
 1 6 2 7 8 8 8 7 5 6 8 6 7 2 2 2 1 8 5 4 0 4 5 7 8 6 7 8 4 1 2 1 8 6 4 2 1
 6 6 6 6 7 7 6 1 1 2 2 6 2 6 8 1 2 6 4 8 6 1 8 6 7 6 1 9 1 2 1 8 4 8 1 2 2
 2 6 6 5 6 6 8 6 2 6 1 9 7 6 6 1 8 8 8 8 2 3 2 5 6 6 5 2 9 2 1 7 1 6 1 8 6
 8 8 6 1 8 2 1 6 6 2 4 9 8 2 8 4 0 8 8 6 2 2 6 6 8 1 0 4 0 2 6 1 2 8 2 2 2
 1 4 4 6 4 4 2 6 0 1 8 8 7 4 2 8 2 1 7 8 7 0 0 9 6 6 1 1 1 8 8 8 8 9 5 7 8
 1 1 6 2 1 1 4 6 2 7 7 1 4 1 2 1 4 6 6 2 8 9 8 8 6 4 8 1 7 0 0 8 7 7 2 0 8
 1 6 2 8 7 2 4 1 1 7 6 6 6 5 7 2 6 7 8 7 4 4 6 7 8 7 7 5 4 6 4 8 2 8 5 1 8
 0 4 2 6 2 9 8 2 1 4 8 8 4 4 2 7 9 4 8 9 2 0 7 1 6 6 8 6 0 8 4 2 7 6 2 9 3
 1 6 7 0 2 5 2 4 7 6 7 0 6 7 1 1 8 6 2 7 2 8 9 8 4 7 2 6 8 7 5 8 5 6 0 0 2
 4 9 8 6 2 5 2 1 1 0 7 8 6 1 8 1 7 6 6 8 4 2 4 9 8 8 9 2 8 4 6 6 5 2 2 0 2
 6 8 2 7 6 2 6 1 5 4 6 8 4 6 7 2 5 1 2 1 4 1 4 8 7 7 6 6 6 4 8 6 6 7 1 8 2
 6 2 7 8 7 8 8 2 2 6 7 8 4 4 5 8 8 1 1 2 8 4 1 0 4 6 1 1 7 8 2 6 6 7 8 4 8
 8 7 2 6 8 5 4 6 6 6 2 6 8 7 1 4 2 0 1 0 8 6 6 5 2 6 6 7 8 2 4 6 2 4 6 4 8
 8 0 7 8 8 4 5 0 2 2 8 7 6 0 1 4 1 2 2 0 1 4 7 8 0 1 4 7 2 9 0 8 2 6 1 1 8
 0 7 1 1 9 4 6 6 0 7 0 2 0 7 8 6 6 8 8 6 8 4 1 4 8 7 7 2 2 2 6 2 2 4 7 1 4
 6 8 1 8 4 1 7 9 2 6 8 1 2 8 2 1 6 8 2 6 2 2 3 8 2 6 4 4 7 1 7 6 8 8 8 7 6
 8 2 1 6 2 1 2 7 8 0 4 2 2 8 2 8 7 6 6 5 9 8 7 9 1 5 8 8 2 4 8 0 9 8 7 6 5
 7 2 9 2 9 4 1 7 8 6 5 0 2 7 2 4 7 9 1 6 5 6 9 6 8 8 4 4 1 1 7 6 1 6 8 1 6
 8 6 6 6 8 1 2 0 7 8 7 2 8 7 7 7 7 2 6 7 6 6 2 2][0m
2021-03-17 22:22:08 Timeit       INFO     [1m[37mTotal Time: 3.09491 sec(s)
[0m
2021-03-17 22:22:08 Main         INFO     [1m[33mAssignment 2 Finished[0m
2021-03-17 22:22:08 Timeit       INFO     [1m[37mFunc: 'main' with args: () took: 19.27430 sec(s)[0m
+ qstat -f 4639149.apollo-acf
