
import os
import pickle

# training_data_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/data_processed_train"
# filename = os.path.join(training_data_path, "processed sample " + str(0) + ".pickle")
# with open(filename, 'rb') as handle:
#     data = pickle.load(handle)

# print(data["eef_traj_1"].shape)

# print(len(os.listdir(training_data_path)))


test_dir_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/data_processed_train"
processed_sample_files = os.listdir(test_dir_path)
print(processed_sample_files)