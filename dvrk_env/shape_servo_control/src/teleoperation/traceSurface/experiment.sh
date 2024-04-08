#!/bin/bash


###########################################################
########## Sample Efficiency Experiment ###################
###########################################################

cd sample_complexity_experiment
python3 loop_process_traj.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/demos_train_straight_flat_2ball --data_processed_root_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/sample_complexity_experiment --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --num_group=30 --num_samples_per_group=30
python3 loop_train.py --data_processed_root_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/sample_complexity_experiment --reward_model_root_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/sample_complexity_experiment/weights --vdp=/home/dvrk/LfD_data/ex_trace_curve_corrected/processed_data_test_straight_flat_2ball
python3 loop_test_and_plot.py --data_processed_root_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/sample_complexity_experiment --reward_model_root_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/sample_complexity_experiment/weights --vdp=/home/dvrk/LfD_data/ex_trace_curve_corrected/processed_data_test_straight_flat_2ball
