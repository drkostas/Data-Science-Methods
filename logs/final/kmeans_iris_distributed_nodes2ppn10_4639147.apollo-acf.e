Can't locate local/lib.pm in @INC (you may need to install the local::lib module) (@INC contains: /nics/b/home/kgeorgio/perl5/lib/perl5 /opt/moab/lib/perl5 /opt/moab/lib/perl5 /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/site_perl/5.26.2/x86_64-linux-thread-multi /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/site_perl/5.26.2 /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/5.26.2/x86_64-linux-thread-multi /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/5.26.2 .).
BEGIN failed--compilation aborted.
+ cd /lustre/haven/proj/UTK0150/kgeorgio/assignment2
+ CONDA_BIN=/nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/
+ conf=/lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment2_isaac_iris_nodes2ppn10.yml
+ log=/lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment2_isaac_iris_nodes2ppn10.log
+ command='/nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment2.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment2_isaac_iris_nodes2ppn10.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment2_isaac_iris_nodes2ppn10.log'
+ eval /nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment2.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment2_isaac_iris_nodes2ppn10.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment2_isaac_iris_nodes2ppn10.log
++ /nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment2.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment2_isaac_iris_nodes2ppn10.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment2_isaac_iris_nodes2ppn10.log
2021-03-17 22:21:49 Main         INFO     [1m[33mStarting Assignment 2[0m
2021-03-17 22:21:49 Config       INFO     [1m[37mConfiguration file loaded successfully[0m
2021-03-17 22:21:49 Main         INFO     [1m[33mInvoking kmeans.py(distributed) for 5 clusters and the iris dataset[0m
2021-03-17 22:21:51 KMeans distributed Proc(0) INFO     [1m[34mStarted with 20 processes.[0m
2021-03-17 22:21:52 KMeans distributed Proc(0) INFO     [1m[34mDataset iris loaded. Shape: (150, 4).[0m
2021-03-17 22:21:52 Timeit       INFO     [1m[37mInit Time: 0.00099 sec(s)[0m
2021-03-17 22:21:52 Timeit       INFO     [1m[37mFirst Loop Distances Calc Time: 0.00011 sec(s)[0m
2021-03-17 22:21:52 Timeit       INFO     [1m[37mFirst Loop Cluster Assignment Time: 0.00002 sec(s)[0m
2021-03-17 22:21:52 Timeit       INFO     [1m[37mFirst Loop Centroids Update Time: 0.00191 sec(s)[0m
2021-03-17 22:21:52 KMeans distributed Proc(0) INFO     [1m[34mFinal Cluster Assignments: 
[4 4 4 4 4 2 4 4 4 4 2 4 4 4 2 2 2 4 2 2 2 2 4 4 4 4 4 2 4 4 4 2 2 2 4 4 2
 4 4 4 4 4 4 4 2 4 2 4 2 4 0 0 0 1 0 1 0 1 0 1 1 1 1 0 1 0 1 1 0 1 0 1 0 0
 0 0 0 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 1 1 1 1 1 1 1 3 0 3 0 3 3 1 3 3 3 0
 0 3 0 0 0 0 3 3 0 3 0 3 0 3 3 0 0 0 3 3 3 0 0 0 3 3 0 0 3 3 0 0 3 3 0 0 0
 0 0][0m
2021-03-17 22:21:52 Timeit       INFO     [1m[37mTotal Time: 0.01159 sec(s)
[0m
2021-03-17 22:21:52 Main         INFO     [1m[33mAssignment 2 Finished[0m
2021-03-17 22:21:52 Timeit       INFO     [1m[37mFunc: 'main' with args: () took: 3.33593 sec(s)[0m
+ qstat -f 4639147.apollo-acf
