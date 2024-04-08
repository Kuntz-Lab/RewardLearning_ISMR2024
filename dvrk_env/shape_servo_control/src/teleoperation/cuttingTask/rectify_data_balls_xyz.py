import argparse
import pickle
import os

'''
multiply the saved balls relative x-y coordinates to the tissue by 2 (just to correct a mistake I made before)
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--rigid_state_path', type=str, help="where rigid state data is")
    parser.add_argument('--ori_urdf_path', type=str, help="where the urdf of the original tissue urdf (not retracted) is")

    args = parser.parse_args()
    rigid_state_path = args.rigid_state_path
    ori_urdf_path = args.ori_urdf_path

    full_data_recording_path = os.path.join(rigid_state_path, "full_data")
    num_groups = len(os.listdir(full_data_recording_path))

    for group_count in range(num_groups):
        with open(os.path.join(full_data_recording_path, f"group {group_count}.pickle"), 'rb') as handle:
            full_data = pickle.load(handle)
        balls_relative_xyz = full_data["balls_relative_xyz"]

        balls_relative_xyz[0:2]*=2
        full_data["balls_relative_xyz"] = balls_relative_xyz

        print(f"multiply by 2 rigid state {group_count}")

        with open(os.path.join(full_data_recording_path, f"group {group_count}.pickle"), 'wb') as handle:
            pickle.dump(full_data, handle, protocol=3)



    for group_count in range(num_groups):
        with open(os.path.join(ori_urdf_path, f"tissue_{group_count}_balls_relative_xyz.pickle"), 'rb') as handle:
            balls_xyz_data = pickle.load(handle)

        balls_relative_xyz = balls_xyz_data["balls_relative_xyz"]
        balls_relative_xyz[0][0] = balls_relative_xyz[0][0]*2
        balls_relative_xyz[0][1] = balls_relative_xyz[0][1]*2
        balls_xyz_data["balls_relative_xyz"] = balls_relative_xyz

        print(f"{group_count} multiply by 2 original object urdf")

        with open(os.path.join(ori_urdf_path, f"tissue_{group_count}_balls_relative_xyz.pickle"), 'wb') as handle:
            pickle.dump(balls_xyz_data, handle, protocol=3)


