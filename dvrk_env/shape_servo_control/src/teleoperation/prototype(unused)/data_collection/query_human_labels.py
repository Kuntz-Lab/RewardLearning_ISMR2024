#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pickle
import timeit

def get_states(data):
    """Return all states of the trajectory"""

    states = []

    for eef_state in data["traj"]:
        states.append(list(eef_state["pose"]["p"]))   # states: shape (traj_length, 3). 

    return np.array(states)


def load_demo_and_image(demo_path, img_path, group, sample):
    file = os.path.join(demo_path, f"group {group} sample {sample}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)      

    image = mpimg.imread(os.path.join(img_path, f"group {group} sample {sample}.png")) 
    
    return data, image

def save_data(label, data_1, data_2, idx, save_path):
    processed_data = {"traj_1": data_1["states"], "traj_2": data_2["states"], \
                    "obj_embedding": data_1["obj_embedding"],   
                    "label": label} 
                    
    
    with open(os.path.join(save_path, "processed sample " + str(idx) + ".pickle"), 'wb') as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)      


image_main_path = "/home/dvrk/LfD_data/group_meeting/images/processed_images" 
demos_path = "/home/dvrk/LfD_data/group_meeting/demos_w_embedding"
processed_demos_path = "/home/dvrk/LfD_data/group_meeting/processed_demos_w_embedding"
os.makedirs(processed_demos_path, exist_ok=True)
num_data_pt = 100
num_samples_per_group = 27 
NUM_GROUP = 1 

# fig = plt.figure()
start_time = timeit.default_timer() 
# for idx in range(num_data_pt):
idx = len(os.listdir(processed_demos_path)) #0
while idx < num_data_pt:

    if idx % 1 == 0:
        print("========================================")
        print("current count:", idx, " , time passed:", timeit.default_timer() - start_time)

    group_idx_0 = np.random.randint(low=0, high=NUM_GROUP)
    sample_idx_0 = np.random.randint(low=0, high=num_samples_per_group)
    group_idx_1 = group_idx_0 #np.random.randint(low=0, high=NUM_GROUP)
    sample_idx_1 = np.random.randint(low=0, high=num_samples_per_group)
    
    data_0, image_0 = load_demo_and_image(demos_path, image_main_path, group_idx_0, sample_idx_0)
    data_1, image_1 = load_demo_and_image(demos_path, image_main_path, group_idx_1, sample_idx_1)
    
    # plt.title("Which trajectory is the best (0 or 1)?")
    
    
    fig, axarr = plt.subplots(1,2, figsize=(8, 4), dpi=150)  #
    axarr[0].imshow(image_0)
    axarr[1].imshow(image_1)
    axarr[0].title.set_text('Trajectory 0')
    axarr[1].title.set_text('Trajectory 1')    
    # ax.set_title('title', fontsize=16)
    axarr[0].axis('off')
    axarr[1].axis('off')
    

    # plt.show()
    plt.draw()
    plt.pause(0.5) 
    
    label = input("Your label (0 or 1): ")
    label = int(label)
    plt.close(fig)
    # print("label:", label)
    
    if label not in [0, 1]:
        print("XXXXXXX Wrong label, ignore this label")
        
    else:        
        print(f"YEAH! Confirmed label: {label}")
        save_data(label, data_0, data_1, idx, processed_demos_path)
        idx += 1
    
    
    
