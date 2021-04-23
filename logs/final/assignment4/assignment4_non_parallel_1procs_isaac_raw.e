Can't locate local/lib.pm in @INC (you may need to install the local::lib module) (@INC contains: /nics/b/home/kgeorgio/perl5/lib/perl5 /opt/moab/lib/perl5 /opt/moab/lib/perl5 /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/site_perl/5.26.2/x86_64-linux-thread-multi /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/site_perl/5.26.2 /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/5.26.2/x86_64-linux-thread-multi /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/5.26.2 .).
BEGIN failed--compilation aborted.
++ head -n 1 /var/spool/torque/aux//4767414.apollo-acf
+ export MASTER_ADDR=acf-sc100
+ MASTER_ADDR=acf-sc100
+ export MASTER_PORT=12345
+ MASTER_PORT=12345
+ cd /lustre/haven/proj/UTK0150/kgeorgio/playground/assignment4
+ CONDA_BIN=/nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/
+ conf=/lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment4_non_parallel_1.yml
+ log=/lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log
+ command='/nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment4.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment4_non_parallel_1.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log'
+ eval /nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment4.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment4_non_parallel_1.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log
++ /nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment4.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment4_non_parallel_1.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log
2021-04-23 15:14:33 FancyLogger  INFO     [1m[37mLogger is set. Log file path: /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log[0m
2021-04-23 15:14:33 Main         INFO     [1m[33mStarting Assignment 4[0m
2021-04-23 15:14:33 Config       INFO     [1m[37mConfiguration file loaded successfully[0m
2021-04-23 15:14:33 FancyLogger  INFO     [1m[37mLogger is set. Log file path: /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4_non_parallel_1nprocs.log[0m
2021-04-23 15:14:33 CnnRunner    INFO     [1m[32mModel parameters are configured.[0m
2021-04-23 15:14:33 CnnRunner    INFO     [1m[32mDevice: cpu[0m
2021-04-23 15:14:33 CnnRunner    INFO     [1m[32mModel Architecture:
LeNet5(
  (conv_1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv_2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (conv_3): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
  (linear_1): Linear(in_features=120, out_features=84, bias=True)
  (linear_2): Linear(in_features=84, out_features=10, bias=True)
)[0m
2021-04-23 15:14:33 CnnRunner    INFO     [1m[32mMnist dataset loaded successfully.[0m
2021-04-23 15:14:33 CnnRunner    INFO     [1m[32mNon-parallel mode with 1 proc(s) requested..[0m

2021-04-23 15:24:19 CnnRunner    INFO     [1m[35mTraining Finished! Results:[0m

2021-04-23 15:24:22 CnnRunner    INFO     [1m[34mTest Loss: 5.834054499864578e-05[0m
2021-04-23 15:24:22 CnnRunner    INFO     [1m[34mCorrect/Total : 9891/10000[0m
2021-04-23 15:24:22 CnnRunner    INFO     [1m[34mAccuracy: 98.91%[0m
2021-04-23 15:24:22 Main         INFO     [1m[33mAssignment 4 Finished[0m
2021-04-23 15:24:22 Timeit       INFO     [1m[37mFunc: 'main' with args: () took: 588.63461 sec(s)[0m
+ qstat -f 4767414.apollo-acf