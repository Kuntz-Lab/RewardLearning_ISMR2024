import os
import timeit
import argparse


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_recording_path', type=str, help="where you want to record data")
    parser.add_argument('--rigid_state_path', type=str, help="root path of the rigid state")
    parser.add_argument('--headless', default="False", type=str, help="run without viewer?")
    parser.add_argument('--save_data', default="False", type=str, help="True: save recorded data to pickles files")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")
    
    args = parser.parse_args()

    num_group = len(os.listdir(os.path.join(args.rigid_state_path, "full_data")))
    print("############## NUM group (ball configs): ", num_group)

    start_time = timeit.default_timer() 
    for i in range(num_group):
        os.system(f"python3 collect_pcd.py --data_recording_path={args.data_recording_path} --rigid_state_path={args.rigid_state_path} --group_count={i} --headless={args.headless} --save_data={args.save_data} --rand_seed={args.rand_seed} --object_name=tissue_{i}")
        
    print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600:.2f} trees" )