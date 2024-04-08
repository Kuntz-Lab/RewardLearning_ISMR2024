#!/bin/bash


####################################################################
# this file contains an example list of commands that you can run #
###################################################################

####################################################################
############################# Reward Learning ######################
###################################################################

####### collect training data and validation data for autoencoder ########
cd data_collection
python3 collect_pcd.py --data_recording_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/demos_train_straight_flat_2ball --headless=True --save_data=True --rand_seed=2021 --num_groups=71000 --max_num_balls=2 --curve_type=2ballFlatLinear
python3 collect_pcd.py --data_recording_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/demos_test_straight_flat_2ball --headless=True --save_data=True --rand_seed=2022 --num_groups=1000 --max_num_balls=2 --curve_type=2ballFlatLinear
cd ..

####### process training data and validation data for autoencoder
cd process_data
python3 process_partial_pc.py --data_recording_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/demos_train_straight_flat_2ball --data_processed_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/processed_data_train_straight_flat_2ball --vis=False
python3 process_partial_pc.py --data_recording_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/demos_test_straight_flat_2ball --data_processed_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/processed_data_test_straight_flat_2ball --vis=False
cd ..

##### train and evaluate autoencoder (remember to select the correct tradeoff constant)
cd pointcloud_representation_learning
python3 training_AE.py --train_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/processed_data_train_straight_flat_2ball --test_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/processed_data_test_straight_flat_2ball --weight_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_small --tradeoff_constant=0.195 --train_len=71000
python3 evaluate_AE.py
cd ..

##############################################################################################

####### collect training and validatioin trajectory data ##########
# you also need to run the motion planning server in another terminal. Source the catkin_ws and Run roslaunch shape_servo_control dvrk_isaac.launch
cd data_collection
python3 collect_curve_traj.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/demos_train_straight_flat_2ball --headless True --save_data True --record_pc True --rand_seed 2021 --num_groups 30 --num_samples 30 --max_num_balls 2 --overlap_tolerance 0.0001 --curve_type=2ballFlatLinear
python3 collect_curve_traj.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/demos_test_straight_flat_2ball --headless True --save_data True --record_pc True --rand_seed 1945 --num_groups 10 --num_samples 10 --max_num_balls 2 --overlap_tolerance 0.0001 --curve_type=2ballFlatLinear
cd ..

###### process training and validation data for reward net 
# remember to modify how you rank the demonstrations (what ground truth reward function, use bonus or not)
cd process_data
python3 process_traj_w_reward.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/demos_train_straight_flat_2ball --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/processed_data_train_straight_flat_2ball --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --num_group=30 --num_samples_per_group=30 --num_data_pt=14000
python3 process_traj_w_reward.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/demos_test_straight_flat_2ball --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/processed_data_test_straight_flat_2ball --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --num_group=10 --num_samples_per_group=10 --num_data_pt=2000
cd ..

###### train and evaluate reward net
cd reward_learning
cd fully_connected
python train.py --rmp=/home/dvrk/LfD_data/ex_trace_curve_corrected/weights_straight_flat_2ball/weights_1 --tdp=/home/dvrk/LfD_data/ex_trace_curve_corrected/processed_data_train_straight_flat_2ball --vdp=/home/dvrk/LfD_data/ex_trace_curve_corrected/processed_data_test_straight_flat_2ball
python test.py --rmp=/home/dvrk/LfD_data/ex_trace_curve_corrected/weights_straight_flat_2ball/weights_1 --tdp=/home/dvrk/LfD_data/ex_trace_curve_corrected/processed_data_train_straight_flat_2ball --test_len=14000

##### visualize ground truth ranking and reward
python3 visualize_gt_ranking.py --demo_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/demos_train_straight_flat_2ball --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/processed_data_train_straight_flat_2ball --num_data_pt=14000 --max_num_balls=2
python3 visualize_gt_reward.py # remember to modify the ball configuration and ground truth reward function including the bonus amount

##### visualize learned ranking and learned reward
python3 visualize_ranking.py --demo_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/demos_train_straight_flat_2ball --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/processed_data_train_straight_flat_2ball --rmp=/home/dvrk/LfD_data/ex_trace_curve_corrected/weights_straight_flat_2ball/weights_1/epoch_200 --num_data_pt=14000 --max_num_balls=2
python3 visualize_reward.py # remember to modify the variables AE_model_path, reward_model_path, vis_data_path and frame
cd ../..

##### collect 4 example trajectories to illustrate preferences (reach both balls, 1 ball and go near another, go near 1 ball, go randomly without reaching any balls)
cd data_collection
python3 collect_ex_traj.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/ex_traj --headless False --save_data True --record_pc True --rand_seed 1945 --max_num_balls 2 --overlap_tolerance 0.0001 --curve_type=2ballFlatLinear
cd ../reward_learning/fully_connected

# remember to modify the titles of the figures and what ground truth reward you are using in this code
python3 visualize_reward_preference --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/ex_traj --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --rmp=/home/dvrk/LfD_data/ex_trace_curve_corrected/weights_straight_flat_2ball/weights_1/epoch_200

#####################################################################################################################
######################################## behvioral cloning ##########################################################
#####################################################################################################################
cd behavioral_cloning/pointcloud_pos_control

# process train and test traj into states and rewards
python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/demos_train_straight_flat_2ball --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_demos_train_BC --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --num_groups=30 --num_samples_per_group=30 --rand_seed=2021
python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/demos_test_straight_flat_2ball --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_demos_test_BC --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --num_groups=10 --num_samples_per_group=10 --rand_seed=1945

#### data1: get state-action pairs from the most preferred trajectories in a total ranking over trajectories
python3 process_total_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_demos_train_BC --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_train_BC_total --num_groups=30 --num_samples_per_group=30 --percent=20 --max_num_balls=2 --rand_seed=2021
python3 process_total_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_demos_test_BC --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_test_BC_total --num_groups=10 --num_samples_per_group=10 --percent=20 --max_num_balls=2 --rand_seed=1945

#### data2: get state-action pairs from the more preferred trajectories in each pairwise trajectory ranking
python3 process_pair_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/processed_data_train_straight_flat_2ball --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_train_BC_pair --max_num_balls=2 --rand_seed=2021
python3 process_pair_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/processed_data_test_straight_flat_2ball --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_test_BC_pair --max_num_balls=2 --rand_seed=1945

# train BC policy on data1 and data2 respectively # remember to change the batch size and the name of the plot in the code
python3 train_actor.py --mp=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/weights_BC_total --tdp=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_train_BC_total --vdp=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_test_BC_total
python3 train_actor.py --mp=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/weights_BC_pair --tdp=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_train_BC_pair --vdp=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_test_BC_pair

# test it on training demo 
python3 test_actor.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC_result_traj --headless False --save_data False --record_pc True --rand_seed 2021 --num_groups 30 --num_samples 1 --max_num_balls 2 --overlap_tolerance 0.0001 --curve_type=2ballFlatLinear --BC_model_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/weights_BC_total/epoch_200 --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --demo_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/demos_train_straight_flat_2ball --use_demo=True
python3 test_actor.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC_result_traj --headless False --save_data False --record_pc True --rand_seed 2021 --num_groups 30 --num_samples 1 --max_num_balls 2 --overlap_tolerance 0.0001 --curve_type=2ballFlatLinear --BC_model_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/weights_BC_pair/epoch_200 --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --demo_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/demos_train_perfect --use_demo=True


# vis
python3 visualize_BC_traj.py --BC_demo_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_demos_train_BC --BC_data_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_train_BC_total

#### train BC on optimal trajectory (collect it first)

######## Note: you can modify collect_perfect_traj.py to either generate random or deterministic init eef pos in move_near_workspace method
python3 collect_perfect_traj.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/demos_train_perfect --headless True --save_data True --record_pc True --rand_seed 2021 --num_groups 900 --num_samples 1 --max_num_balls 2 --overlap_tolerance 0.0001 --curve_type=2ballFlatLinear
python3 collect_perfect_traj.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/demos_test_perfect --headless True --save_data True --record_pc True --rand_seed 1945 --num_groups 100 --num_samples 1 --max_num_balls 2 --overlap_tolerance 0.0001 --curve_type=2ballFlatLinear

python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/demos_train_perfect --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_demos_train_perfect --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --num_groups=900 --num_samples_per_group=1 --rand_seed=2021
python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/demos_test_perfect --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_demos_test_perfect --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --num_groups=100 --num_samples_per_group=1 --rand_seed=1945

python3 process_total_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_demos_train_perfect --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_train_BC_perfect --num_groups=900 --num_samples_per_group=1 --percent=100 --max_num_balls=2 --rand_seed=2021
python3 process_total_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_demos_test_perfect --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_test_BC_perfect --num_groups=10 --num_samples_per_group=1 --percent=100 --max_num_balls=2 --rand_seed=1945

# remember to change the batch size and the name of the plot in the code
python3 train_actor.py --mp=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/weights_BC_perfect --tdp=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_train_BC_perfect --vdp=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_test_BC_perfect
python3 test_actor.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC_result_traj --headless False --save_data False --record_pc True --rand_seed 2021 --num_groups 5 --num_samples 1 --max_num_balls 2 --overlap_tolerance 0.0001 --curve_type=2ballFlatLinear --BC_model_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/weights_BC_perfect/epoch_200 --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --demo_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/demos_train_perfect --use_demo=True


#### train BC on optimal trajectory with varied initial starting pos 
python3 collect_perfect_traj.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/demos_train_perfect_varied --headless True --save_data True --record_pc True --rand_seed 2021 --num_groups 900 --num_samples 1 --max_num_balls 2 --overlap_tolerance 0.0001 --curve_type=2ballFlatLinear
python3 collect_perfect_traj.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/demos_test_perfect_varied --headless True --save_data True --record_pc True --rand_seed 1945 --num_groups 100 --num_samples 1 --max_num_balls 2 --overlap_tolerance 0.0001 --curve_type=2ballFlatLinear

python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/demos_train_perfect_varied --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_demos_train_perfect_varied --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --num_groups=900 --num_samples_per_group=1 --rand_seed=2021
python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/demos_test_perfect_varied --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_demos_test_perfect_varied --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --num_groups=100 --num_samples_per_group=1 --rand_seed=1945

python3 process_total_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_demos_train_perfect_varied --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_train_BC_perfect_varied --num_groups=900 --num_samples_per_group=1 --percent=100 --max_num_balls=2 --rand_seed=2021
python3 process_total_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_demos_test_perfect_varied --data_processed_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_test_BC_perfect_varied --num_groups=100--num_samples_per_group=1 --percent=100 --max_num_balls=2 --rand_seed=1945

# remember to change the batch size and the name of the plot in the code
python3 train_actor.py --mp=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/weights_BC_perfect_varied --tdp=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_train_BC_perfect_varied --vdp=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/processed_data_test_BC_perfect_varied
python3 test_actor.py --data_recording_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC_result_traj --headless False --save_data False --record_pc True --rand_seed 2021 --num_groups 5 --num_samples 1 --max_num_balls 2 --overlap_tolerance 0.0001 --curve_type=2ballFlatLinear --BC_model_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/weights_BC_perfect_varied/epoch_200 --AE_model_path=/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150 --demo_path=/home/dvrk/LfD_data/ex_trace_curve_corrected/BC/demos_train_perfect_varied --use_demo=True

cd ..

python3 reward_mean_and_sigma.py --reward_model_path="/home/dvrk/LfD_data/ex_trace_curve_corrected/weights_straight_flat_2ball/weights_1/epoch_200" --AE_model_path="/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150" --demo_data_root_path="/home/dvrk/LfD_data/ex_trace_curve_corrected/demos_train_straight_flat_2ball" --num_group=30 --num_sample_per_group=30