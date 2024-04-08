#!/usr/bin/env python3
import sys
import os
import timeit



# pkg_path = "/home/dvrk/dvrk_ws"
# os.chdir(pkg_path)



start_time = timeit.default_timer() 

main_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_2/"
reward_model_paths = []
reward_model_paths.append(os.path.join(main_path, 'weights_3'))
reward_model_paths.append(os.path.join(main_path, 'weights_4'))

training_data_paths = []
training_data_paths.append(os.path.join(main_path, 'processed_data'))
training_data_paths.append(os.path.join(main_path, 'processed_data_2'))

for i in range(len(training_data_paths)):

    os.system(f"python3 train.py --rmp {reward_model_paths[i]} --tdp {training_data_paths[i]}")





print("Elapsed time evaluate ", timeit.default_timer() - start_time)





