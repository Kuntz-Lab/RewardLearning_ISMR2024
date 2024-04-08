import shutil
import os
import random

train_data_dir = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/data_processed_train"
files = os.listdir(train_data_dir)
total_num_data = len(files)
num_data = 100
subset_data_dir = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/{num_data}_data_processed_train"
os.makedirs(subset_data_dir, exist_ok=True)

for i in range(num_data):
    filename = random.choice(files)
    file_path = os.path.join(train_data_dir, filename)
    files.remove(filename)
    new_filename = f"processed sample {i}.pickle"
    shutil.copyfile(file_path, os.path.join(subset_data_dir, new_filename))
    print(f"copied file {filename} to {subset_data_dir}; renamed to {new_filename}")
