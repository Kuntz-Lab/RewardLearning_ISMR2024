
import os
import numpy as np
import pickle
import timeit
import matplotlib.pyplot as plt
from PIL import Image

import os
import moviepy.video.io.ImageSequenceClip

'''
plot data across episodes and make video for the last episode of training
'''

def make_video(image_folder_path, image_filenames, save_path, fps=1):
    image_files = [os.path.join(image_folder_path, img)
               for img in image_filenames
               if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(save_path)

def get_episode_data(data_recording_path, episode_idx):
    file = os.path.join(data_recording_path, f"episode {episode_idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return data

# def plot_curves(xs, ys1, ys2, ys3, title, path, x_label, y_label, curve_labels):
#     fig, ax = plt.subplots()
#     ax.plot(xs, ys1, label=curve_labels[0])
#     ax.plot(xs, ys2, label=curve_labels[1])
#     ax.plot(xs, ys3, label=curve_labels[2])
#     ax.legend()
#     ax.set_title(title)
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     os.makedirs(path, exist_ok=True)
#     fig.savefig(f"{path}/{title}.png")
#     plt.cla()
#     plt.close(fig)

def plot_curves(xs, ys_list, title, path, x_label, y_label, curve_labels):
    fig, ax = plt.subplots()
    assert(len(curve_labels)==len(ys_list))
    for idx in range(len(ys_list)):
        ax.plot(xs, ys_list[idx], label=curve_labels[idx])
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{title}.png")
    plt.cla()
    plt.close(fig)    
    



if __name__ == "__main__":
    ### CHANGE ####
    root_folder = "/home/dvrk/RL_data/dvrkControlTest"
    reward_folder = "eef_vel_very_fast_step_2"
    data_recording_path = f"{root_folder}/{reward_folder}/data"
    num_episodes = 2
    last_episode_idx = num_episodes - 1
    step = 10
    print(f"num episodes: {num_episodes}")
    video_path = f"{root_folder}/{reward_folder}/videos/videos_{last_episode_idx}"
    plot_path = f"{root_folder}/{reward_folder}"
    os.makedirs(video_path, exist_ok=True)

    plot_data = True
    get_video = True
    
    ############ images in the episode for 1 env
    data = get_episode_data(data_recording_path, episode_idx=last_episode_idx)

    if plot_data:
        ################## plot eef vel per timestep in each episode
        eef_vels = data["eef_vels"]
        if len(eef_vels)!=0:
            timesteps = [i for i in range(len(eef_vels))]
            print("num eef vels: ", len(timesteps))
            num_envs = eef_vels[0].shape[0]
            plot_path = os.path.join(f"{root_folder}/{reward_folder}", f"episode {last_episode_idx}", "eef_vel")
            os.makedirs(plot_path, exist_ok=True)

            for i in range(num_envs):
                env_eef_vels = [eef_vels[t][i][0:3] for t in timesteps]
                x_eef_vels = [eef_vel[0] for eef_vel in env_eef_vels]
                y_eef_vels = [eef_vel[1] for eef_vel in env_eef_vels]
                z_eef_vels = [eef_vel[2] for eef_vel in env_eef_vels]
                plot_curves(xs=timesteps, ys_list=[x_eef_vels, y_eef_vels, z_eef_vels], title=f"{reward_folder} eef vel in env {i}", \
                            x_label="timestep", y_label="value of eef vel in each component", curve_labels=["x_vel", "y_vel", "z_vel"], path=plot_path)
                
        ################## plot eef pos per timestep in each episode
        eef_poses = data["eef_poses"]
        timesteps = [i for i in range(len(eef_poses))]
        print("num eef poses: ", len(timesteps))
        num_envs = eef_poses[0].shape[0]
        plot_path = os.path.join(f"{root_folder}/{reward_folder}", f"episode {last_episode_idx}", "eef_pos")
        os.makedirs(plot_path, exist_ok=True)

        
        for i in range(num_envs):
            env_eef_poses = [eef_poses[t][i][0:3] for t in timesteps]
            x_eef_poses = [eef_pos[0] for eef_pos in env_eef_poses]
            y_eef_poses = [eef_pos[1] for eef_pos in env_eef_poses]
            z_eef_poses = [eef_pos[2] for eef_pos in env_eef_poses]
            plot_curves(xs=timesteps, ys_list=[x_eef_poses, y_eef_poses, z_eef_poses], title=f"{reward_folder} env {i}", \
                            x_label="timestep", y_label="value of eef pos in each component", curve_labels=["x_pos", "y_pos", "z_pos"], path=plot_path)

    if get_video:
        for video_env in range(num_envs):
            images = data["images"]
            images_env_0 = [im[video_env, :, :, :] for im in images]
            print(f"num images in one episode: {len(images_env_0)}")

            for env_idx, im in enumerate(images_env_0):
                processed_image = Image.fromarray(im)
                processed_image.save(os.path.join(video_path, f"img_{env_idx}.png"))

            ######### make video ##########
            print("making video from images...")
            image_filenames = [f"img_{i}.png" for i in range(len(images_env_0))]
            make_video(video_path, image_filenames, save_path=f'{root_folder}/{reward_folder}/rl_video_{last_episode_idx}_env_{video_env}.mp4' , fps=2)