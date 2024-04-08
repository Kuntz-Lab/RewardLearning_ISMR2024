#!/bin/bash

####################################################################
# this file contains an example list of commands that you can run #
###################################################################

############# Autoencoder ############################
python3 collect_pcd.py --data_recording_path=/home/dvrk/LfD_data/ex_AE_boxCone/demos_train --headless=True --save_data=True --rand_seed=2021 --num_samples=51000
python3 collect_pcd.py --data_recording_path=/home/dvrk/LfD_data/ex_AE_boxCone/demos_test --headless=True --save_data=True --rand_seed=1945 --num_samples=1000

python3 process_partial_pc.py --data_recording_path=/home/dvrk/LfD_data/ex_AE_boxCone/demos_train --data_processed_path=/home/dvrk/LfD_data/ex_AE_boxCone/processed_data_train --vis=False
python3 process_partial_pc.py --data_recording_path=/home/dvrk/LfD_data/ex_AE_boxCone/demos_test --data_processed_path=/home/dvrk/LfD_data/ex_AE_boxCone/processed_data_test --vis=False

cd pointcloud_representation_learning
python3 training_AE.py --train_path=/home/dvrk/LfD_data/ex_AE_boxCone/processed_data_train --test_path=/home/dvrk/LfD_data/ex_AE_boxCone/processed_data_test --weight_path=/home/dvrk/LfD_data/ex_AE_boxCone/weights/weights_1 --tradeoff_constant=0.18 --train_len=51000
python3 evaluate_AE.py
cd ..

############# trajectory data #######################
python3 collect_push_traj_pointmass.py --data_recording_path=/home/dvrk/LfD_data/ex_push/demos_train --headless True --save_data True --record_pc True --rand_seed 2021 --num_groups 30 --num_samples 30 --overlap_tolerance 0.001 --is_behavioral_cloning False
python3 collect_push_traj_pointmass.py --data_recording_path=/home/dvrk/LfD_data/ex_push/demos_test --headless True --save_data True --record_pc True --rand_seed 1945 --num_groups 10 --num_samples 10 --overlap_tolerance 0.001 --is_behavioral_cloning False

############# Reward learning #####################
cd reward_learning
python3 process_traj.py --data_recording_path=/home/dvrk/LfD_data/ex_push/demos_train --data_processed_path=/home/dvrk/LfD_data/ex_push/data_processed_train --AE_model_path=/home/dvrk/LfD_data/ex_AE_boxCone/weights/weights_1/epoch_150 --num_group=30 --num_samples_per_group=30 --num_data_pt=14000
python3 process_traj.py --data_recording_path=/home/dvrk/LfD_data/ex_push/demos_test --data_processed_path=/home/dvrk/LfD_data/ex_push/data_processed_test --AE_model_path=/home/dvrk/LfD_data/ex_AE_boxCone/weights/weights_1/epoch_150 --num_group=10 --num_samples_per_group=10 --num_data_pt=1400

python3 train.py --rmp=/home/dvrk/LfD_data/ex_push/weights/weights_1 --tdp=/home/dvrk/LfD_data/ex_push/data_processed_train --vdp=/home/dvrk/LfD_data/ex_push/data_processed_test
python3 test.py --rmp=/home/dvrk/LfD_data/ex_push/weights/weights_1 --tdp=/home/dvrk/LfD_data/ex_push/data_processed_test --test_len=1400
python3 test.py --rmp=/home/dvrk/LfD_data/ex_push/weights/weights_1 --tdp=/home/dvrk/LfD_data/ex_push/data_processed_train --test_len=14000

###### visualize reward
python3 collect_visualize_reward.py --data_recording_path=/home/dvrk/LfD_data/ex_push/vis_reward --headless=False --save_data=True --rand_seed=1357 --num_samples=700 --cone_xy -0.05 -0.51
# remember to change the title of each plot in the code
python3 visualize_reward.py --rmp=/home/dvrk/LfD_data/ex_push/weights/weights_1/epoch_100 --AE_model_path=/home/dvrk/LfD_data/ex_AE_boxCone/weights/weights_1/epoch_150 --vis_data_path=/home/dvrk/LfD_data/ex_push/vis_reward/sample_0.pickle 
cd ..

############# Behavioral cloning ##########################
python3 collect_push_traj_pointmass.py --data_recording_path=/home/dvrk/LfD_data/ex_push/BC/demos_train --headless True --save_data True --record_pc True --rand_seed 2021 --num_groups 50 --num_samples 1 --overlap_tolerance 0.001 --is_behavioral_cloning True
python3 collect_push_traj_pointmass.py --data_recording_path=/home/dvrk/LfD_data/ex_push/BC/demos_test --headless True --save_data True --record_pc True --rand_seed 1945 --num_groups 20 --num_samples 1 --overlap_tolerance 0.001 --is_behavioral_cloning True

cd behavioral_cloning

# pointcloud embedding and pos action
cd pointcloud_pos_control
python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_push/BC/demos_train --data_processed_path=/home/dvrk/LfD_data/ex_push/BC/pointcloud_pos_control/data_processed_train --AE_model_path=/home/dvrk/LfD_data/ex_AE_boxCone/weights/weights_1/epoch_150 --num_group=50
python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_push/BC/demos_test --data_processed_path=/home/dvrk/LfD_data/ex_push/BC/pointcloud_pos_control/data_processed_test --AE_model_path=/home/dvrk/LfD_data/ex_AE_boxCone/weights/weights_1/epoch_150 --num_group=10

python3 train_actor.py --mp=/home/dvrk/LfD_data/ex_push/BC/pointcloud_pos_control/weights/weights_1 --tdp=/home/dvrk/LfD_data/ex_push/BC/pointcloud_pos_control/data_processed_train --vdp=/home/dvrk/LfD_data/ex_push/BC/pointcloud_pos_control/data_processed_test

python3 test_actor_pointmass.py --headless False --data_recording_path=/home/dvrk/LfD_data/ex_push/BC/debug --model_path=/home/dvrk/LfD_data/ex_push/BC/pointcloud_pos_control/weights/weights_1/epoch_200 --AE_model_path=/home/dvrk/LfD_data/ex_AE_boxCone/weights/weights_1/epoch_150 --rand_seed=2021
cd ..

# cartesian coordinate and pos action
cd pos_control
python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_push/BC/demos_train --data_processed_path=/home/dvrk/LfD_data/ex_push/BC/pos_control/data_processed_train --num_group=50
python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_push/BC/demos_test --data_processed_path=/home/dvrk/LfD_data/ex_push/BC/pos_control/data_processed_test --num_group=10

python3 train_actor.py --mp=/home/dvrk/LfD_data/ex_push/BC/pos_control/weights/weights_1 --tdp=/home/dvrk/LfD_data/ex_push/BC/pos_control/data_processed_train --vdp=/home/dvrk/LfD_data/ex_push/BC/pos_control/data_processed_test

python3 test_actor_pointmass.py --headless False --data_recording_path=/home/dvrk/LfD_data/ex_push/BC/debug --model_path=/home/dvrk/LfD_data/ex_push/BC/pos_control/weights/weights_1/epoch_200 --rand_seed=2021
cd ..



