import os
import timeit
import argparse
import pickle
import shutil
import random

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_recording_path', type=str, help="path to recorded data")
    parser.add_argument('--AE_model_path', type=str, help="path to pre-trained autoencoder weights")
    parser.add_argument('--data_processed_root_path', type=str, help="path to root folder containing processed data (where to save the data collected during the loop)")
    parser.add_argument('--num_group', default=30, type=int, help="num groups to process")
    parser.add_argument('--num_samples_per_group', default=30, type=int, help="num samples per group")
    
    args = parser.parse_args()
    num_group = args.num_group
    num_sample_per_group = args.num_samples_per_group
    data_recording_path = args.data_recording_path
    AE_model_path = args.AE_model_path
    data_processed_root_path = args.data_processed_root_path

    os.makedirs(data_processed_root_path, exist_ok=True)

    max_num_pairs = int(num_group * (num_sample_per_group*(num_sample_per_group-1)/2))
    print("max_num_pairs", max_num_pairs)
    num_data_list = []
    curr_num_data = max_num_pairs
    while curr_num_data >= 256:
        num_data_list.append(int(curr_num_data))
        curr_num_data = int(curr_num_data/2)

    print("############## num_data_list: ", num_data_list)

    with open(os.path.join(data_processed_root_path, "num_data_list.pickle"), 'wb') as handle:
            pickle.dump(num_data_list, handle, protocol=3)     


    # start_time = timeit.default_timer() 
    # for num_data in num_data_list:
    #     data_processed_path = os.path.join(data_processed_root_path, f"processed_data_train_{num_data}")
    #     os.system(f"python3 ../process_data/process_traj_w_reward_no_repeat.py --data_recording_path={data_recording_path} --data_processed_path={data_processed_path} --AE_model_path={AE_model_path} --num_group={num_group} --num_samples_per_group={num_sample_per_group} --num_data_pt={num_data}")
        
    # print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600:.2f} trees" )

    # use process_traj_all_pairs.py instead of process_traj_w_reward_no_repeat.py
    data_processed_path = os.path.join(data_processed_root_path, f"processed_data_train_{max_num_pairs}")
    os.system(f"python3 ../process_data/process_traj_all_pairs.py --data_recording_path={data_recording_path} --data_processed_path={data_processed_path} --AE_model_path={AE_model_path} --num_group={num_group} --num_samples_per_group={num_sample_per_group}")
    
    all_preferences = os.listdir(data_processed_path)

    for i in range(1, len(num_data_list), 1):
        
        num_data = num_data_list[i]
        data_processed_path = os.path.join(data_processed_root_path, f"processed_data_train_{num_data}")
        os.makedirs(data_processed_path, exist_ok=True)

        rand_preferences = random.sample(all_preferences, num_data)
        data_idx = 0
        for preference in rand_preferences:
            src = os.path.join(data_processed_root_path, f"processed_data_train_{max_num_pairs}", preference)
            dest = os.path.join(data_processed_root_path, f"processed_data_train_{num_data}", f"processed sample {data_idx}.pickle")
            shutil.copyfile(src, dest)
            data_idx+=1
