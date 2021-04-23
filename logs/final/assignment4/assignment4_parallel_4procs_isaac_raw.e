Can't locate local/lib.pm in @INC (you may need to install the local::lib module) (@INC contains: /nics/b/home/kgeorgio/perl5/lib/perl5 /opt/moab/lib/perl5 /opt/moab/lib/perl5 /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/site_perl/5.26.2/x86_64-linux-thread-multi /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/site_perl/5.26.2 /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/5.26.2/x86_64-linux-thread-multi /nics/e/sw/cs400_centos7.3_acfsoftware/anaconda3/4.4.0/centos7.3_gnu6.3.0/anaconda3-4.4.0/lib/5.26.2 .).
BEGIN failed--compilation aborted.
++ head -n 1 /var/spool/torque/aux//4767417.apollo-acf
+ export MASTER_ADDR=acf-sc101
+ MASTER_ADDR=acf-sc101
+ export MASTER_PORT=34567
+ MASTER_PORT=34567
+ cd /lustre/haven/proj/UTK0150/kgeorgio/playground/assignment4
+ CONDA_BIN=/nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/
+ conf=/lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment4_parallel_4.yml
+ log=/lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log
+ command='/nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment4.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment4_parallel_4.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log'
+ eval /nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment4.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment4_parallel_4.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log
++ /nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python assignment4.py -c /lustre/haven/proj/UTK0150/kgeorgio/playground/confs/assignment4_parallel_4.yml -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log
2021-04-23 15:27:46 FancyLogger  INFO     [1m[37mLogger is set. Log file path: /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4.log[0m
2021-04-23 15:27:46 Main         INFO     [1m[33mStarting Assignment 4[0m
2021-04-23 15:27:46 Config       INFO     [1m[37mConfiguration file loaded successfully[0m
2021-04-23 15:27:46 Main         INFO     [1m[33m//nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/mpirun -n 4 /nics/b/home/kgeorgio/.conda/envs/dse512_playground/bin/python /lustre/haven/proj/UTK0150/kgeorgio/playground/assignment4/cnn_runner.py --dataset-name mnist --dataset-path ../data/mnist -ep 30 -btr 128 -bte 1000 -lr 0.05 -mo 0.5 -l /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4_data_parallel_4nprocs.log [0m
2021-04-23 15:27:49 FancyLogger  INFO     [1m[37mLogger is set. Log file path: /lustre/haven/proj/UTK0150/kgeorgio/playground/logs/assignment4_data_parallel_4nprocs.log[0m
2021-04-23 15:27:49 CnnRunner    INFO     [1m[32mModel parameters are configured.[0m
2021-04-23 15:27:49 CnnRunner    INFO     [1m[32mDevice: cpu[0m
2021-04-23 15:27:49 CnnRunner    INFO     [1m[32mModel Architecture:
LeNet5(
  (conv_1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv_2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (conv_3): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
  (linear_1): Linear(in_features=120, out_features=84, bias=True)
  (linear_2): Linear(in_features=84, out_features=10, bias=True)
)[0m
2021-04-23 15:27:49 CnnRunner    INFO     [1m[32mMnist dataset loaded successfully.[0m
2021-04-23 15:27:49 CnnRunner    INFO     [1m[32mData Parallel mode with 4 proc(s) requested..[0m
2021-04-23 15:27:49 CnnRunner    INFO     [1m[32mWorld size: 4[0m
Training Epochs:   0%|          | 0/30 [00:00<?, ?it/s]Training Epochs:   0%|          | 0/30 [00:06<?, ?it/s, epoch_accuracy=0.628, epoch_loss=1.13, epoch_time=6.55]Training Epochs:   3%|▎         | 1/30 [00:06<03:09,  6.55s/it, epoch_accuracy=0.628, epoch_loss=1.13, epoch_time=6.55]Training Epochs:   3%|▎         | 1/30 [00:12<03:09,  6.55s/it, epoch_accuracy=0.93, epoch_loss=0.234, epoch_time=6.17]Training Epochs:   7%|▋         | 2/30 [00:12<02:57,  6.32s/it, epoch_accuracy=0.93, epoch_loss=0.234, epoch_time=6.17]Training Epochs:   7%|▋         | 2/30 [00:18<02:57,  6.32s/it, epoch_accuracy=0.955, epoch_loss=0.151, epoch_time=6.2]Training Epochs:  10%|█         | 3/30 [00:18<02:49,  6.27s/it, epoch_accuracy=0.955, epoch_loss=0.151, epoch_time=6.2]Training Epochs:  10%|█         | 3/30 [00:25<02:49,  6.27s/it, epoch_accuracy=0.967, epoch_loss=0.108, epoch_time=6.18]Training Epochs:  13%|█▎        | 4/30 [00:25<02:42,  6.23s/it, epoch_accuracy=0.967, epoch_loss=0.108, epoch_time=6.18]Training Epochs:  13%|█▎        | 4/30 [00:31<02:42,  6.23s/it, epoch_accuracy=0.977, epoch_loss=0.0782, epoch_time=6.17]Training Epochs:  17%|█▋        | 5/30 [00:31<02:35,  6.21s/it, epoch_accuracy=0.977, epoch_loss=0.0782, epoch_time=6.17]Training Epochs:  17%|█▋        | 5/30 [00:37<02:35,  6.21s/it, epoch_accuracy=0.982, epoch_loss=0.0598, epoch_time=6.18]Training Epochs:  20%|██        | 6/30 [00:37<02:28,  6.20s/it, epoch_accuracy=0.982, epoch_loss=0.0598, epoch_time=6.18]Training Epochs:  20%|██        | 6/30 [00:43<02:28,  6.20s/it, epoch_accuracy=0.985, epoch_loss=0.0465, epoch_time=6.19]Training Epochs:  23%|██▎       | 7/30 [00:43<02:22,  6.20s/it, epoch_accuracy=0.985, epoch_loss=0.0465, epoch_time=6.19]Training Epochs:  23%|██▎       | 7/30 [00:50<02:22,  6.20s/it, epoch_accuracy=0.987, epoch_loss=0.0381, epoch_time=6.41]Training Epochs:  27%|██▋       | 8/30 [00:50<02:17,  6.27s/it, epoch_accuracy=0.987, epoch_loss=0.0381, epoch_time=6.41]Training Epochs:  27%|██▋       | 8/30 [00:56<02:17,  6.27s/it, epoch_accuracy=0.99, epoch_loss=0.0338, epoch_time=6.5]  Training Epochs:  30%|███       | 9/30 [00:56<02:13,  6.34s/it, epoch_accuracy=0.99, epoch_loss=0.0338, epoch_time=6.5]Training Epochs:  30%|███       | 9/30 [01:03<02:13,  6.34s/it, epoch_accuracy=0.992, epoch_loss=0.0235, epoch_time=6.48]Training Epochs:  33%|███▎      | 10/30 [01:03<02:07,  6.38s/it, epoch_accuracy=0.992, epoch_loss=0.0235, epoch_time=6.48]Training Epochs:  33%|███▎      | 10/30 [01:09<02:07,  6.38s/it, epoch_accuracy=0.992, epoch_loss=0.0258, epoch_time=6.48]Training Epochs:  37%|███▋      | 11/30 [01:09<02:01,  6.41s/it, epoch_accuracy=0.992, epoch_loss=0.0258, epoch_time=6.48]Training Epochs:  37%|███▋      | 11/30 [01:15<02:01,  6.41s/it, epoch_accuracy=0.993, epoch_loss=0.0238, epoch_time=6.11]Training Epochs:  40%|████      | 12/30 [01:15<01:53,  6.32s/it, epoch_accuracy=0.993, epoch_loss=0.0238, epoch_time=6.11]Training Epochs:  40%|████      | 12/30 [01:21<01:53,  6.32s/it, epoch_accuracy=0.997, epoch_loss=0.0117, epoch_time=6.13]Training Epochs:  43%|████▎     | 13/30 [01:21<01:46,  6.26s/it, epoch_accuracy=0.997, epoch_loss=0.0117, epoch_time=6.13]Training Epochs:  43%|████▎     | 13/30 [01:27<01:46,  6.26s/it, epoch_accuracy=0.997, epoch_loss=0.0116, epoch_time=6.13]Training Epochs:  47%|████▋     | 14/30 [01:27<01:39,  6.22s/it, epoch_accuracy=0.997, epoch_loss=0.0116, epoch_time=6.13]Training Epochs:  47%|████▋     | 14/30 [01:34<01:39,  6.22s/it, epoch_accuracy=0.993, epoch_loss=0.0239, epoch_time=6.16]Training Epochs:  50%|█████     | 15/30 [01:34<01:33,  6.21s/it, epoch_accuracy=0.993, epoch_loss=0.0239, epoch_time=6.16]Training Epochs:  50%|█████     | 15/30 [01:40<01:33,  6.21s/it, epoch_accuracy=0.993, epoch_loss=0.0237, epoch_time=6.61]Training Epochs:  53%|█████▎    | 16/30 [01:40<01:28,  6.33s/it, epoch_accuracy=0.993, epoch_loss=0.0237, epoch_time=6.61]Training Epochs:  53%|█████▎    | 16/30 [01:47<01:28,  6.33s/it, epoch_accuracy=0.997, epoch_loss=0.0099, epoch_time=6.64]Training Epochs:  57%|█████▋    | 17/30 [01:47<01:23,  6.42s/it, epoch_accuracy=0.997, epoch_loss=0.0099, epoch_time=6.64]Training Epochs:  57%|█████▋    | 17/30 [01:53<01:23,  6.42s/it, epoch_accuracy=0.999, epoch_loss=0.00439, epoch_time=6.6]Training Epochs:  60%|██████    | 18/30 [01:53<01:17,  6.48s/it, epoch_accuracy=0.999, epoch_loss=0.00439, epoch_time=6.6]Training Epochs:  60%|██████    | 18/30 [02:00<01:17,  6.48s/it, epoch_accuracy=0.999, epoch_loss=0.00508, epoch_time=6.57]Training Epochs:  63%|██████▎   | 19/30 [02:00<01:11,  6.51s/it, epoch_accuracy=0.999, epoch_loss=0.00508, epoch_time=6.57]Training Epochs:  63%|██████▎   | 19/30 [02:07<01:11,  6.51s/it, epoch_accuracy=1, epoch_loss=0.00219, epoch_time=6.59]    Training Epochs:  67%|██████▋   | 20/30 [02:07<01:05,  6.53s/it, epoch_accuracy=1, epoch_loss=0.00219, epoch_time=6.59]Training Epochs:  67%|██████▋   | 20/30 [02:13<01:05,  6.53s/it, epoch_accuracy=0.999, epoch_loss=0.00323, epoch_time=6.61]Training Epochs:  70%|███████   | 21/30 [02:13<00:59,  6.56s/it, epoch_accuracy=0.999, epoch_loss=0.00323, epoch_time=6.61]Training Epochs:  70%|███████   | 21/30 [02:20<00:59,  6.56s/it, epoch_accuracy=0.999, epoch_loss=0.00278, epoch_time=6.62]Training Epochs:  73%|███████▎  | 22/30 [02:20<00:52,  6.58s/it, epoch_accuracy=0.999, epoch_loss=0.00278, epoch_time=6.62]Training Epochs:  73%|███████▎  | 22/30 [02:26<00:52,  6.58s/it, epoch_accuracy=1, epoch_loss=0.00141, epoch_time=6.57]    Training Epochs:  77%|███████▋  | 23/30 [02:26<00:46,  6.57s/it, epoch_accuracy=1, epoch_loss=0.00141, epoch_time=6.57]Training Epochs:  77%|███████▋  | 23/30 [02:33<00:46,  6.57s/it, epoch_accuracy=1, epoch_loss=0.000737, epoch_time=6.62]Training Epochs:  80%|████████  | 24/30 [02:33<00:39,  6.59s/it, epoch_accuracy=1, epoch_loss=0.000737, epoch_time=6.62]Training Epochs:  80%|████████  | 24/30 [02:40<00:39,  6.59s/it, epoch_accuracy=1, epoch_loss=0.000717, epoch_time=6.64]Training Epochs:  83%|████████▎ | 25/30 [02:40<00:33,  6.60s/it, epoch_accuracy=1, epoch_loss=0.000717, epoch_time=6.64]Training Epochs:  83%|████████▎ | 25/30 [02:46<00:33,  6.60s/it, epoch_accuracy=1, epoch_loss=0.00131, epoch_time=6.61] Training Epochs:  87%|████████▋ | 26/30 [02:46<00:26,  6.61s/it, epoch_accuracy=1, epoch_loss=0.00131, epoch_time=6.61]Training Epochs:  87%|████████▋ | 26/30 [02:53<00:26,  6.61s/it, epoch_accuracy=1, epoch_loss=0.00034, epoch_time=6.65]Training Epochs:  90%|█████████ | 27/30 [02:53<00:19,  6.62s/it, epoch_accuracy=1, epoch_loss=0.00034, epoch_time=6.65]Training Epochs:  90%|█████████ | 27/30 [03:00<00:19,  6.62s/it, epoch_accuracy=1, epoch_loss=0.000269, epoch_time=6.62]Training Epochs:  93%|█████████▎| 28/30 [03:00<00:13,  6.62s/it, epoch_accuracy=1, epoch_loss=0.000269, epoch_time=6.62]Training Epochs:  93%|█████████▎| 28/30 [03:06<00:13,  6.62s/it, epoch_accuracy=1, epoch_loss=0.000225, epoch_time=6.54]Training Epochs:  97%|█████████▋| 29/30 [03:06<00:06,  6.60s/it, epoch_accuracy=1, epoch_loss=0.000225, epoch_time=6.54]Testing: 0it [00:00, ?it/s]Testing: 0it [00:00, ?it/s, test_loss_accum=0.12]Testing: 1it [00:00,  3.49it/s, test_loss_accum=0.12]Testing: 0it [00:00, ?it/s]Testing: 1it [00:00,  3.49it/s, test_loss_accum=0.245]Testing: 2it [00:00,  3.61it/s, test_loss_accum=0.245]Testing: 0it [00:00, ?it/s, test_loss_accum=0.119]Testing: 1it [00:00,  3.41it/s, test_loss_accum=0.119]Testing: 2it [00:00,  3.61it/s, test_loss_accum=0.322]Testing: 3it [00:00,  3.66it/s, test_loss_accum=0.322]Testing: 1it [00:00,  3.41it/s, test_loss_accum=0.179]Testing: 2it [00:00,  3.50it/s, test_loss_accum=0.179]Testing: 3it [00:01,  3.66it/s, test_loss_accum=0.42] Testing: 4it [00:01,  3.67it/s, test_loss_accum=0.42]Testing: 2it [00:00,  3.50it/s, test_loss_accum=0.259]Testing: 3it [00:00,  3.55it/s, test_loss_accum=0.259]Testing: 4it [00:01,  3.67it/s, test_loss_accum=0.524]Testing: 5it [00:01,  3.65it/s, test_loss_accum=0.524]Testing: 3it [00:01,  3.55it/s, test_loss_accum=0.345]Testing: 4it [00:01,  3.57it/s, test_loss_accum=0.345]Testing: 5it [00:01,  3.65it/s, test_loss_accum=0.621]Testing: 6it [00:01,  3.67it/s, test_loss_accum=0.621]Testing: 6it [00:01,  3.67it/s, test_loss_accum=0.722]Testing: 7it [00:01,  3.69it/s, test_loss_accum=0.722]Testing: 4it [00:01,  3.57it/s, test_loss_accum=0.458]Testing: 5it [00:01,  3.60it/s, test_loss_accum=0.458]Testing: 7it [00:02,  3.69it/s, test_loss_accum=0.85] Testing: 8it [00:02,  3.70it/s, test_loss_accum=0.85]Testing: 5it [00:01,  3.60it/s, test_loss_accum=0.528]Testing: 6it [00:01,  3.62it/s, test_loss_accum=0.528]Testing: 8it [00:02,  3.70it/s, test_loss_accum=0.98]Testing: 9it [00:02,  3.67it/s, test_loss_accum=0.98]Testing: 6it [00:01,  3.62it/s, test_loss_accum=0.613]Testing: 7it [00:01,  3.59it/s, test_loss_accum=0.613]Testing: 9it [00:02,  3.67it/s, test_loss_accum=1.07]Testing: 10it [00:02,  3.69it/s, test_loss_accum=1.07]                                                      Testing: 7it [00:02,  3.59it/s, test_loss_accum=0.687]Testing: 8it [00:02,  3.61it/s, test_loss_accum=0.687]Testing: 8it [00:02,  3.61it/s, test_loss_accum=0.816]Testing: 9it [00:02,  3.64it/s, test_loss_accum=0.816]Testing: 9it [00:02,  3.64it/s, test_loss_accum=0.894]Testing: 10it [00:02,  3.66it/s, test_loss_accum=0.894]                                                       Training Epochs:  97%|█████████▋| 29/30 [03:12<00:06,  6.60s/it, epoch_accuracy=1, epoch_loss=0.000198, epoch_time=6.15]Training Epochs: 100%|██████████| 30/30 [03:12<00:00,  6.46s/it, epoch_accuracy=1, epoch_loss=0.000198, epoch_time=6.15]Training Epochs: 100%|██████████| 30/30 [03:12<00:00,  6.42s/it, epoch_accuracy=1, epoch_loss=0.000198, epoch_time=6.15]
2021-04-23 15:31:02 CnnRunner    INFO     [1m[35mTraining Finished! Results:[0m
Testing: 0it [00:00, ?it/s]Testing: 0it [00:00, ?it/s, test_loss_accum=0.109]Testing: 1it [00:00,  3.54it/s, test_loss_accum=0.109]Testing: 1it [00:00,  3.54it/s, test_loss_accum=0.183]Testing: 2it [00:00,  3.69it/s, test_loss_accum=0.183]Testing: 2it [00:00,  3.69it/s, test_loss_accum=0.272]Testing: 3it [00:00,  3.74it/s, test_loss_accum=0.272]Testing: 3it [00:01,  3.74it/s, test_loss_accum=0.357]Testing: 4it [00:01,  3.78it/s, test_loss_accum=0.357]Testing: 4it [00:01,  3.78it/s, test_loss_accum=0.46] Testing: 5it [00:01,  3.76it/s, test_loss_accum=0.46]Testing: 5it [00:01,  3.76it/s, test_loss_accum=0.542]Testing: 6it [00:01,  3.78it/s, test_loss_accum=0.542]Testing: 6it [00:01,  3.78it/s, test_loss_accum=0.584]Testing: 7it [00:01,  3.80it/s, test_loss_accum=0.584]Testing: 7it [00:02,  3.80it/s, test_loss_accum=0.648]Testing: 8it [00:02,  3.82it/s, test_loss_accum=0.648]Testing: 8it [00:02,  3.82it/s, test_loss_accum=0.773]Testing: 9it [00:02,  3.79it/s, test_loss_accum=0.773]Testing: 9it [00:02,  3.79it/s, test_loss_accum=0.825]Testing: 10it [00:02,  3.81it/s, test_loss_accum=0.825]                                                       2021-04-23 15:31:04 CnnRunner    INFO     [1m[34mTesting Finished! Storing results..[0m
2021-04-23 15:31:04 CnnRunner    INFO     [1m[34mTest Loss: 8.252705000340939e-05[0m
2021-04-23 15:31:04 CnnRunner    INFO     [1m[34mCorrect/Total : 9838/10000[0m
2021-04-23 15:31:04 CnnRunner    INFO     [1m[34mAccuracy: 98.38%[0m
Testing: 0it [00:00, ?it/s]Testing: 0it [00:00, ?it/s, test_loss_accum=0.169]Testing: 1it [00:00,  2.93it/s, test_loss_accum=0.169]Testing: 1it [00:00,  2.93it/s, test_loss_accum=0.28] Testing: 2it [00:00,  3.00it/s, test_loss_accum=0.28]Testing: 2it [00:00,  3.00it/s, test_loss_accum=0.365]Testing: 3it [00:00,  3.03it/s, test_loss_accum=0.365]Testing: 3it [00:01,  3.03it/s, test_loss_accum=0.466]Testing: 4it [00:01,  3.06it/s, test_loss_accum=0.466]Testing: 4it [00:01,  3.06it/s, test_loss_accum=0.586]Testing: 5it [00:01,  3.05it/s, test_loss_accum=0.586]Testing: 5it [00:01,  3.05it/s, test_loss_accum=0.685]Testing: 6it [00:01,  3.07it/s, test_loss_accum=0.685]Testing: 6it [00:02,  3.07it/s, test_loss_accum=0.747]Testing: 7it [00:02,  3.08it/s, test_loss_accum=0.747]Testing: 7it [00:02,  3.08it/s, test_loss_accum=0.819]Testing: 8it [00:02,  3.09it/s, test_loss_accum=0.819]Testing: 8it [00:02,  3.09it/s, test_loss_accum=0.95] Testing: 9it [00:02,  3.08it/s, test_loss_accum=0.95]Testing: 9it [00:03,  3.08it/s, test_loss_accum=1.02]Testing: 10it [00:03,  3.12it/s, test_loss_accum=1.02]                                                      2021-04-23 15:31:25 Main         INFO     [1m[33mAssignment 4 Finished[0m
2021-04-23 15:31:25 Timeit       INFO     [1m[37mFunc: 'main' with args: () took: 219.27189 sec(s)[0m
+ qstat -f 4767417.apollo-acf
