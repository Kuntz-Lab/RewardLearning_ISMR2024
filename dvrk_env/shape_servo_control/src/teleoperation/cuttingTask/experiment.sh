#!/bin/bash


###########################################################
########## Sample Efficiency Experiment ###################
###########################################################

cd sample_complexity_experiment
python3 loop_process_traj.py --data_recording_path=/home/dvrk/LfD_data/ex_cut/1ball/demos_train --data_processed_root_path=/home/dvrk/LfD_data/ex_cut/1ball/sample_complexity_experiment --AE_model_path=/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200 --num_group=60 --num_samples_per_group=30
python3 loop_train.py --data_processed_root_path=/home/dvrk/LfD_data/ex_cut/1ball/sample_complexity_experiment --reward_model_root_path=/home/dvrk/LfD_data/ex_cut/1ball/sample_complexity_experiment/weights --vdp=/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_30samples_14000
python3 loop_test_and_plot.py --data_processed_root_path=/home/dvrk/LfD_data/ex_cut/1ball/sample_complexity_experiment --reward_model_root_path=/home/dvrk/LfD_data/ex_cut/1ball/sample_complexity_experiment/weights --vdp=/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_30samples_14000
