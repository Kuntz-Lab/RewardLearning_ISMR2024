#!/bin/bash


####################################################################
# this file contains an example list of commands that you can run #
###################################################################


####################################################################################
############## collect meshes and urdf for train & test data of autoencoder ########
####################################################################################
#conda activate py39
cd deformable_utils
python3 create_meshes.py --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh"
python3 create_urdfs.py --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --object_urdf_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/cutting_urdf_train" --rand_seed=2023 --num_config=10000 --max_num_balls=1
python3 create_urdfs.py --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --object_urdf_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/cutting_urdf_test" --rand_seed=1997 --num_config=1000 --max_num_balls=1
cd ..

# conda deactivate
# conda activate rlgpu
cd data_collection
python3 loop_collect_rigid_state.py --data_recording_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/rigid_state_train" --object_urdf_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/cutting_urdf_train" --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --headless=True --save_data True --rand_seed=2023
python3 loop_collect_rigid_state.py --data_recording_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/rigid_state_test" --object_urdf_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/cutting_urdf_test" --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --headless=True --save_data True --rand_seed=1997
python3 loop_collect_pcd.py --data_recording_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/processed_data_train" --rigid_state_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/rigid_state_train" --headless=True --save_data True --rand_seed=2023
python3 loop_collect_pcd.py --data_recording_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/processed_data_test" --rigid_state_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/rigid_state_test" --headless=True --save_data True --rand_seed=1997
cd ..

# cd process_data
# python3 process_partial_pc.py --data_recording_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/data_train" --data_processed_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/processed_data_train" --vis=True
# python3 process_partial_pc.py --data_recording_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/data_test" --data_processed_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/processed_data_test" --vis=True
# cd ..

# conda deactivate
# conda activate emd_7
# I use tradeoff constant 0.179 for pointconv, 0.175 for conv1d with 10000 data
cd pointcloud_representation_learning
python3 training_AE.py --train_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/processed_data_train" --test_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/processed_data_test" --weight_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_1" --tradeoff_constant=0.175 --train_len=10000
python3 evaluate_AE.py
cd ..

######################################################################################
###### collect meshes and urdf for train & test demo of reward learning ##############
#######################################################################################
# conda activate py39
cd deformable_utils
python3 create_urdfs.py --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --object_urdf_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_urdf_train" --rand_seed=2021 --num_config=60 --max_num_balls=1
python3 create_urdfs.py --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --object_urdf_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_urdf_test" --rand_seed=1945 --num_config=20 --max_num_balls=1
cd ..

# conda deactivate
# conda activate rlgpu
cd data_collection
python3 loop_collect_rigid_state.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/rigid_state_train" --object_urdf_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_urdf_train" --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --headless=True --save_data True --rand_seed=2021
python3 loop_collect_rigid_state.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/rigid_state_test" --object_urdf_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_urdf_test" --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --headless=False --save_data True --rand_seed=1945

# you also need to run the motion planning server in another terminal. Source the catkin_ws and Run roslaunch shape_servo_control dvrk_isaac.launch
python3 loop_collect_cut_traj.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train" --rigid_state_path="/home/dvrk/LfD_data/ex_cut/1ball/rigid_state_train" --num_samples=60 --overlap_tolerance=0.0001 --headless=True --save_data=True --rand_seed=2021
python3 loop_collect_cut_traj.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_test" --rigid_state_path="/home/dvrk/LfD_data/ex_cut/1ball/rigid_state_test" --num_samples=20 --overlap_tolerance=0.0001 --headless=False --save_data=True --rand_seed=1945
cd ..

# conda deactivate
# conda activate py39
cd process_data
python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_inv_dist" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --num_group=60 --num_samples_per_group=30 --num_data_pt=14000 --rand_seed=2021
python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_test" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_inv_dist" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --num_group=20 --num_samples_per_group=20 --num_data_pt=2600 --rand_seed=1945
cd ..

cd reward_learning
cd fully_connected
python train.py --rmp="/home/dvrk/LfD_data/ex_cut/1ball/weights_inv_dist" --tdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_inv_dist" --vdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_inv_dist"
python test.py --rmp="/home/dvrk/LfD_data/ex_cut/1ball/weights_inv_dist/epoch_400" --tdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_inv_dist" --test_len=14000

##### visualize ground truth ranking and reward
python3 visualize_gt_ranking.py --demo_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_inv_dist" --num_data_pt=14000 --max_num_balls=1

##### visualize learned ranking and learned reward
python3 visualize_ranking.py --demo_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_inv_dist" --rmp="/home/dvrk/LfD_data/ex_cut/1ball/weights_inv_dist/epoch_300" --num_data_pt=14000 --max_num_balls=1
python3 visualize_reward.py --reward_model_path="/home/dvrk/LfD_data/ex_cut/1ball/weights_inv_dist/epoch_400" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --vis_data_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train/group 0 sample 0.pickle"
cd ../..


###### prepare rigid state of the retracted tissue for RL
cd deformable_utils
python3 create_urdfs.py --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --object_urdf_path="/home/dvrk/LfD_data/RL_cut/1ball/cutting_urdf" --rand_seed=2000 --num_config=1000 --max_num_balls=1
cd ../data_collection
python3 loop_collect_rigid_state.py --data_recording_path="/home/dvrk/LfD_data/RL_cut/1ball/rigid_state" --object_urdf_path="/home/dvrk/LfD_data/RL_cut/1ball/cutting_urdf" --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --headless=True --save_data True --rand_seed=2000
cd ..



# ## Miscellaneous (can ignore): ##############################
# python3 loop_collect_cut_traj.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/deletethis" --rigid_state_path="/home/dvrk/LfD_data/ex_cut/1ball/rigid_state_train" --num_samples=60 --overlap_tolerance=0.0001 --headless=True --save_data=True --rand_seed=2021
#Just for me messing up the attachment point coordinates
# python3 rectify_data_balls_xyz.py --rigid_state_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/rigid_state_test --ori_urdf_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/cutting_urdf_test
# python3 rectify_data_balls_xyz.py --rigid_state_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/rigid_state_train --ori_urdf_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/cutting_urdf_train
# python3 rectify_data_balls_xyz.py --rigid_state_path=/home/dvrk/LfD_data/RL_cut/1ball/rigid_state --ori_urdf_path=/home/dvrk/LfD_data/RL_cut/1ball/cutting_urdf

# # 60 groups, 30 samples, 14000 pref
# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_30samples_14000" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --num_group=60 --num_samples_per_group=30 --num_data_pt=14000 --rand_seed=2021
# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_test" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_30samples_14000" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --num_group=20 --num_samples_per_group=20 --num_data_pt=1400 --rand_seed=1945

# python train.py --rmp="/home/dvrk/LfD_data/ex_cut/1ball/weights_30samples_14000" --tdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_30samples_14000" --vdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_30samples_14000" --plot_category="30samples_14000"
# python test.py --rmp="/home/dvrk/LfD_data/ex_cut/1ball/weights_30samples_14000/epoch_400" --tdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_30samples_14000" --test_len=14000
# python3 visualize_reward.py --reward_model_path="/home/dvrk/LfD_data/ex_cut/1ball/weights_30samples_14000/epoch_400" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --vis_data_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train/group 0 sample 0.pickle"

# #### try pointconv to see if it improves
# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_pointconv" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_pointconv/epoch_200" --num_group=60 --num_samples_per_group=30 --num_data_pt=14000 --rand_seed=2021
# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_test" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_pointconv" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_pointconv/epoch_200" --num_group=20 --num_samples_per_group=20 --num_data_pt=1400 --rand_seed=1945

# python train.py --rmp="/home/dvrk/LfD_data/ex_cut/1ball/weights_pointconv" --tdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_pointconv" --vdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_pointconv" --plot_category="pointconv"
# python test.py --rmp="/home/dvrk/LfD_data/ex_cut/1ball/weights_pointconv/epoch_400" --tdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_30samples_14000" --test_len=14000
# python3 visualize_reward.py --reward_model_path="/home/dvrk/LfD_data/ex_cut/1ball/weights_pointconv/epoch_400" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_pointconv/epoch_200" --vis_data_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train/group 0 sample 0.pickle"




# # 60 groups, 60 samples, 14000 pref
# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_60samples_14000" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --num_group=60 --num_samples_per_group=60 --num_data_pt=14000 --rand_seed=2021
# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_test" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_60samples_14000" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --num_group=20 --num_samples_per_group=20 --num_data_pt=1400 --rand_seed=1945

# python train.py --rmp="/home/dvrk/LfD_data/ex_cut/1ball/weights_60samples_14000" --tdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_60samples_14000" --vdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_60samples_14000" --plot_category="60samples_14000"
# python3 visualize_reward.py --reward_model_path="/home/dvrk/LfD_data/ex_cut/1ball/weights_60samples_14000/epoch_400" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --vis_data_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train/group 0 sample 0.pickle"


# # 60 groups, 60 samples 26000 pref
# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_60samples_26000" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --num_group=60 --num_samples_per_group=60 --num_data_pt=26000 --rand_seed=2021
# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_test" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_60samples_26000" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --num_group=20 --num_samples_per_group=20 --num_data_pt=2600 --rand_seed=1945

# python train.py --rmp="/home/dvrk/LfD_data/ex_cut/1ball/weights_60samples_26000" --tdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_60samples_26000" --vdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_60samples_26000" --plot_category="60samples_26000"

# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_no_bon" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --num_group=60 --num_samples_per_group=30 --num_data_pt=14000 --rand_seed=2021
# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_test" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_no_bon" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --num_group=20 --num_samples_per_group=20 --num_data_pt=1400 --rand_seed=1945

# python train.py --rmp="/home/dvrk/LfD_data/ex_cut/1ball/weights_no_bon" --tdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_no_bon" --vdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_no_bon" --plot_category="no_bon"
# python3 visualize_reward.py --reward_model_path="/home/dvrk/LfD_data/ex_cut/1ball/weights_60samples_26000/epoch_400" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --vis_data_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train/group 0 sample 0.pickle"

########################################################
######## last minute fix
########################################################
# python3 loop_collect_cut_traj.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train_change" --rigid_state_path="/home/dvrk/LfD_data/ex_cut/1ball/rigid_state_train" --num_samples=30 --overlap_tolerance=0.000025 --headless=True --save_data=True --rand_seed=2021
# python3 loop_collect_cut_traj.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_test_change" --rigid_state_path="/home/dvrk/LfD_data/ex_cut/1ball/rigid_state_test" --num_samples=20 --overlap_tolerance=0.000025 --headless=False --save_data=True --rand_seed=1945

# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train_change" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_change" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --num_group=60 --num_samples_per_group=30 --num_data_pt=500 --rand_seed=2021

# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train_change" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_change_pointconv" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_pointconv/epoch_200" --num_group=60 --num_samples_per_group=30 --num_data_pt=14000 --rand_seed=2021
# python train.py --rmp="/home/dvrk/LfD_data/ex_cut/1ball/weights_change_pointconv" --tdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_change_pointconv" --vdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_change" --plot_category="change_pointconv"

# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_test_change" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_change" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --num_group=20 --num_samples_per_group=20 --num_data_pt=1400 --rand_seed=1945
# python train.py --rmp="/home/dvrk/LfD_data/ex_cut/1ball/weights_change" --tdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_change" --vdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_change" --plot_category="change"

# python3 visualize_gt_ranking.py --demo_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train_change" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_change" --num_data_pt=14000 --max_num_balls=1
# python3 visualize_reward.py --reward_model_path="/home/dvrk/LfD_data/ex_cut/1ball/weights_change/epoch_400" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --vis_data_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train_change/group 0 sample 0.pickle"


# python3 process_traj_w_reward.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train_change" --data_processed_path="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_change_bonus" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --num_group=60 --num_samples_per_group=30 --num_data_pt=14000 --rand_seed=2021

# python train.py --rmp="/home/dvrk/LfD_data/ex_cut/1ball/weights_change_bonus" --tdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_change_bonus" --vdp="/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_change" --plot_category="change_bonus"

########################################
###### Behaviroal cloning ##############
########################################

# conda activate py39
cd deformable_utils
python3 create_urdfs.py --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --object_urdf_path="/home/dvrk/LfD_data/ex_cut/1ball/BC/cutting_urdf_train" --rand_seed=2021 --num_config=1800 --max_num_balls=1
python3 create_urdfs.py --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --object_urdf_path="/home/dvrk/LfD_data/ex_cut/1ball/BC/cutting_urdf_test" --rand_seed=1945 --num_config=400 --max_num_balls=1
cd ..

cd behavioral_cloning/pointcloud_pos_control

# process train and test traj into states and rewards
python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_cut/1ball/demos_train --data_processed_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_demos_train_BC --AE_model_path=/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200 --num_groups=60 --num_samples_per_group=30 --rand_seed=2021
python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_cut/1ball/demos_test --data_processed_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_demos_test_BC --AE_model_path=/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200 --num_groups=20 --num_samples_per_group=20 --rand_seed=1945

#### data1: get state-action pairs from the most preferred trajectories in a total ranking over trajectories
python3 process_total_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_demos_train_BC --data_processed_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_data_train_BC_total --num_groups=60 --num_samples_per_group=30 --percent=20 --max_num_balls=1 --rand_seed=2021
python3 process_total_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_demos_test_BC --data_processed_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_data_test_BC_total --num_groups=20 --num_samples_per_group=20 --percent=20 --max_num_balls=1 --rand_seed=1945

#### data2: get state-action pairs from the more preferred trajectories in each pairwise trajectory ranking
python3 process_pair_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_cut/1ball/processed_data_train_30samples_14000 --data_processed_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_data_train_BC_pair --max_num_balls=1 --rand_seed=2021
python3 process_pair_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_cut/1ball/processed_data_test_30samples_14000 --data_processed_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_data_test_BC_pair --max_num_balls=1 --rand_seed=1945

# train BC policy on data1 and data2 respectively # remember to change the batch size and the name of the plot in the code
python3 train_actor.py --mp=/home/dvrk/LfD_data/ex_cut/1ball/BC/weights_BC_total --tdp=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_data_train_BC_total --vdp=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_data_test_BC_total
python3 train_actor.py --mp=/home/dvrk/LfD_data/ex_cut/1ball/BC/weights_BC_pair --tdp=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_data_train_BC_pair --vdp=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_data_test_BC_pair


# conda deactivate
# conda activate rlgpu
cd ../../data_collection
python3 loop_collect_rigid_state.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/BC/rigid_state_train" --object_urdf_path="/home/dvrk/LfD_data/ex_cut/1ball/BC/cutting_urdf_train" --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --headless=True --save_data True --rand_seed=2021
python3 loop_collect_rigid_state.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/BC/rigid_state_test" --object_urdf_path="/home/dvrk/LfD_data/ex_cut/1ball/BC/cutting_urdf_test" --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --headless=True --save_data True --rand_seed=1945
cd ../behavioral_cloning/pointcloud_pos_control

# you also need to run the motion planning server in another terminal. Source the catkin_ws and Run roslaunch shape_servo_control dvrk_isaac.launch
python3 loop_collect_perfect_traj.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/BC/demos_train_perfect" --rigid_state_path="/home/dvrk/LfD_data/ex_cut/1ball/BC/rigid_state_train" --num_samples=1 --overlap_tolerance=0.0001 --headless=True --save_data=True --rand_seed=2021
python3 loop_collect_perfect_traj.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/1ball/BC/demos_test_perfect" --rigid_state_path="/home/dvrk/LfD_data/ex_cut/1ball/BC/rigid_state_test" --num_samples=1 --overlap_tolerance=0.0001 --headless=True --save_data=True --rand_seed=1945

python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/demos_train_perfect --data_processed_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_demos_train_perfect --AE_model_path=/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200 --num_groups=3600 --num_samples_per_group=1 --rand_seed=2021
python3 process_traj_BC.py --data_recording_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/demos_test_perfect --data_processed_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_demos_test_perfect --AE_model_path=/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200 --num_groups=400 --num_samples_per_group=1 --rand_seed=1945

python3 process_total_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_demos_train_perfect --data_processed_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_data_train_BC_perfect --num_groups=1800 --num_samples_per_group=1 --percent=100 --max_num_balls=1 --rand_seed=2021
python3 process_total_rank.py --data_recording_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_demos_test_perfect --data_processed_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_data_test_BC_perfect --num_groups=400 --num_samples_per_group=1 --percent=100 --max_num_balls=1 --rand_seed=1945

python3 train_actor.py --mp=/home/dvrk/LfD_data/ex_cut/1ball/BC/weights_BC_perfect --tdp=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_data_train_BC_perfect --vdp=/home/dvrk/LfD_data/ex_cut/1ball/BC/processed_data_test_BC_perfect

cd ..

#### miscellaneous
# python3 test_actor.py --data_recording_path=/home/dvrk/LfD_data/deletethis --rigid_state_path="/home/dvrk/LfD_data/ex_cut/1ball/BC/rigid_state_train" --group_count=0 --num_samples=1 --overlap_tolerance=0.0001 --headless=False --save_data=False --rand_seed=2021 --object_name=tissue_0 --BC_model_path=/home/dvrk/LfD_data/ex_cut/1ball/BC/weights_BC_perfect/epoch_200 --AE_model_path=/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200


###### get sample mean and std of rewards
python3 reward_mean_and_sigma.py --reward_model_path="/home/dvrk/LfD_data/ex_cut/1ball/weights_30samples_14000/epoch_400" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --demo_data_root_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train" --num_group=60 
python3 reward_mean_and_sigma.py --reward_model_path="/home/dvrk/LfD_data/ex_cut/1ball/sample_complexity_experiment/weights/weights_815/epoch_400" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --demo_data_root_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train" --num_group=60 
python3 reward_mean_and_sigma.py --reward_model_path="/home/dvrk/LfD_data/ex_cut/1ball/weights_change/epoch_400" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --demo_data_root_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train_change" --num_group=60 

python3 visualize_reward.py --reward_model_path="/home/dvrk/LfD_data/ex_cut/1ball/sample_complexity_experiment/weights/weights_815/epoch_400" --AE_model_path="/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200" --vis_data_path="/home/dvrk/LfD_data/ex_cut/1ball/demos_train/group 0 sample 0.pickle"



##### random stuff
python3 loop_collect_pcd.py --data_recording_path="/home/dvrk/LfD_data/RL_cut/1ball/PC" --rigid_state_path="/home/dvrk/LfD_data/RL_cut/1ball/rigid_state" --headless=True --save_data True --rand_seed=2023

python3 create_urdfs.py --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --object_urdf_path="/home/dvrk/LfD_data/ex_cut/debug/cutting_urdf_train" --rand_seed=2023 --num_config=1 --max_num_balls=1
python3 loop_collect_rigid_state.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/debug/rigid" --object_urdf_path="/home/dvrk/LfD_data/ex_cut/debug/cutting_urdf_train" --object_mesh_path="/home/dvrk/LfD_data/ex_cut/1ball/cutting_mesh" --headless=False --save_data True --rand_seed=2023
python3 loop_collect_pcd.py --data_recording_path="/home/dvrk/LfD_data/ex_cut/debug/pc" --rigid_state_path="/home/dvrk/LfD_data/ex_cut/debug/rigid" --headless=False --save_data True --rand_seed=2023
