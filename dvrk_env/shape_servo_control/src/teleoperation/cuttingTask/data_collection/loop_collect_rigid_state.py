import os
import timeit
import argparse
import sys


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_recording_path', type=str, help="where you want to record data")
    parser.add_argument('--object_urdf_path', type=str, help="root path of the urdf")
    parser.add_argument('--object_mesh_path', type=str, help="root path of the meshes and primitive dict")
    parser.add_argument('--headless', default="False", type=str, help="Index of grasp candidate to test")
    parser.add_argument('--save_data', default="True", type=str, help="True: save recorded data to pickles files")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")
    
    args = parser.parse_args()

    num_tissue = len(os.listdir(args.object_urdf_path))//2
    print("############## NUM Tissues (ball configs): ", num_tissue)

    if args.save_data == "False":
        print("You will fall into an infinite loop with save_data == False")
        sys.exit("Exiting the code with sys.exit()!")

    os.makedirs(args.data_recording_path, exist_ok=True)
    os.makedirs(os.path.join(args.data_recording_path, "full_data"), exist_ok=True)
    os.makedirs(os.path.join(args.data_recording_path, "tri_mesh"), exist_ok=True)
    os.makedirs(os.path.join(args.data_recording_path, "urdf"), exist_ok=True)

    start_time = timeit.default_timer() 
    i = len(os.listdir(os.path.join(args.data_recording_path, "full_data")))
    while i < num_tissue:
        print(f"== $$$$$$$$$$$ get rigid state for tissue_{i} $$$$$$$$$$$$ ==")
        os.system(f"python3 collect_rigid_state.py --data_recording_path={args.data_recording_path} --object_urdf_path={args.object_urdf_path} --object_mesh_path={args.object_mesh_path} --headless={args.headless} --save_data={args.save_data} --rand_seed={args.rand_seed} --object_name=tissue_{i}")
        i = len(os.listdir(os.path.join(args.data_recording_path, "full_data")))
        
    print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600:.2f} trees" )