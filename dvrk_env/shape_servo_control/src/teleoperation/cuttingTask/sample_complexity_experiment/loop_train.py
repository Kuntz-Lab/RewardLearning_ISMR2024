import os
import timeit
import argparse
import pickle


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_processed_root_path', type=str, help="path to root folder containing processed data")
    parser.add_argument('--reward_model_root_path', type=str, help="path to root folder containing reward model")
    parser.add_argument('--vdp', type=str, help="path to validation data")

    args = parser.parse_args()
    data_processed_root_path = args.data_processed_root_path
    reward_model_root_path = args.reward_model_root_path
    vdp = args.vdp

    with open(os.path.join(data_processed_root_path, "num_data_list.pickle"), 'rb') as handle:
        num_data_list = pickle.load(handle)

    print("############## num_data_list: ", num_data_list)

    start_time = timeit.default_timer() 
    for num_data in num_data_list:
        num_data = int(num_data)
        rmp = os.path.join(reward_model_root_path, f"weights_{num_data}")
        tdp = os.path.join(data_processed_root_path, f"processed_data_train_{num_data}")
        os.system(f"python ../reward_learning/fully_connected/train.py --rmp={rmp} --tdp={tdp} --vdp={vdp} --plot_category={num_data}")
        
    print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600:.2f} trees" )