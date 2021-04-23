Can't locate local/lib.pm in @INC (you may need to install the local::lib module) (@INC contains: /nics/b/home/kgeorgio/perl5/lib/perl5 /opt/moab/lib/perl5 /opt/moab/lib/perl5 /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/site_perl/5.26.2/x86_64-linux-thread-multi /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/site_perl/5.26.2 /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/5.26.2/x86_64-linux-thread-multi /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/5.26.2 .).
BEGIN failed--compilation aborted.
++ head -n 1 /var/spool/torque/aux//4767418.apollo-acf
+ export MASTER_ADDR=acf-sc101
+ MASTER_ADDR=acf-sc101
+ export MASTER_PORT=23456
+ MASTER_PORT=23456
+ cd /lustre/haven/proj/UTK0150/kgeorgio/playground/assignment4
+ CONDA_BIN=/nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/
+ conf=/lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment4_parallel_8.yml
+ log=/lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log
+ command='/nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment4.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment4_parallel_8.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log'
+ eval /nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment4.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment4_parallel_8.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log
++ /nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment4.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment4_parallel_8.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log
2021-04-23 15:31:44 FancyLogger  INFO     [1m[37mLogger is set. Log file path: /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log[0m
2021-04-23 15:31:44 Main         INFO     [1m[33mStarting Assignment 4[0m
2021-04-23 15:31:44 Config       INFO     [1m[37mConfiguration file loaded successfully[0m
2021-04-23 15:31:44 Main         INFO     [1m[33m//nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/mpirun -n 8 /nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python /lustre/haven/proj/UTK0150/kgeorgio/playground/assignment4/cnn_runner.py --dataset-name mnist --dataset-path ../data/mnist -ep 30 -btr 128 -bte 1000 -lr 0.05 -mo 0.5 -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4_data_parallel_8nprocs.log [0m
2021-04-23 15:31:47 FancyLogger  INFO     [1m[37mLogger is set. Log file path: /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4_data_parallel_8nprocs.log[0m
2021-04-23 15:31:47 CnnRunner    INFO     [1m[32mModel parameters are configured.[0m
2021-04-23 15:31:47 CnnRunner    INFO     [1m[32mDevice: cpu[0m
2021-04-23 15:31:47 CnnRunner    INFO     [1m[32mModel Architecture:
LeNet5(
  (conv_1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv_2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (conv_3): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
  (linear_1): Linear(in_features=120, out_features=84, bias=True)
  (linear_2): Linear(in_features=84, out_features=10, bias=True)
)[0m
2021-04-23 15:31:48 CnnRunner    INFO     [1m[32mMnist dataset loaded successfully.[0m
2021-04-23 15:31:48 CnnRunner    INFO     [1m[32mData Parallel mode with 8 proc(s) requested..[0m
2021-04-23 15:31:48 CnnRunner    INFO     [1m[32mWorld size: 8[0m

2021-04-23 15:33:28 CnnRunner    INFO     [1m[35mTraining Finished! Results:[0m

2021-04-23 15:33:32 CnnRunner    INFO     [1m[34mTest Loss: 0.00010594925358891487[0m
2021-04-23 15:33:32 CnnRunner    INFO     [1m[34mCorrect/Total : 9793/10000[0m
2021-04-23 15:33:32 CnnRunner    INFO     [1m[34mAccuracy: 97.93%[0m

2021-04-23 15:33:48 Timeit       INFO     [1m[37mFunc: 'main' with args: () took: 123.78368 sec(s)[0m
+ qstat -f 4767418.apollo-acf