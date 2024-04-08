import pickle
import os

data_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_indicator_0and1/demos_train"
num_group = 20
num_sample = 20
counts = {"2":0, "4":0, "6":0, "3":0}

for group in range(num_group):
    for sample in range(num_sample):
        file = os.path.join(data_path, f"group {group} sample {sample}.pickle")
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
        traj_type =  data["traj seg label"]
        if traj_type == 2:
            counts["2"]+=1
        elif traj_type == 4:
            counts["4"]+=1
        elif traj_type == 6:
            counts["6"]+=1
        elif traj_type == 3:
            counts["3"]+=1
        else:
            print("error")

print ("traj_stat:", counts)