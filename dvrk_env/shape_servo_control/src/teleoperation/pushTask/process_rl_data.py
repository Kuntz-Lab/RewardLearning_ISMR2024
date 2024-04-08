
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

def plot_curve(xs, ys, title, path, x_label, y_label):
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    #ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{title}.png")
    plt.cla()
    plt.close(fig)    


if __name__ == "__main__":
    ### CHANGE ####
    root_folder = "/home/dvrk/RL_data/dvrkPush"
    reward_folder = "gt_reward_eef_pos"
    data_recording_path = f"{root_folder}/{reward_folder}/data"
    num_episodes = 2971 #len(os.listdir(data_recording_path))
    last_episode_idx = num_episodes - 1
    step = 10
    print(f"num episodes: {num_episodes}")
    video_path = f"{root_folder}/{reward_folder}/videos/videos_{last_episode_idx}"
    plot_path = f"{root_folder}/{reward_folder}"
    os.makedirs(video_path, exist_ok=True)

    plot_data = True
    record_video = True

   

    ######### plot rewards and save images #############
    if plot_data:
        start_time = timeit.default_timer() 
        avg_gt_rewards = []
        avg_learned_rewards = []
        avg_num_reaches = []
        percent_num_env_success = []
        for env_idx in range(0, num_episodes, step):

            if env_idx % 50 == 0:
                print("========================================")
                print("current episode:", env_idx, " , time passed:", timeit.default_timer() - start_time)
            
            data = get_episode_data(data_recording_path, episode_idx=env_idx)
            
            gt_rewards = data["gt_cum_reward"]
            avg_gt_reward = np.sum(gt_rewards)/len(gt_rewards)
            avg_gt_rewards.append(avg_gt_reward)

            learned_rewards = data["learned_cum_reward"]
            avg_learned_reward = np.sum(learned_rewards)/len(learned_rewards)
            avg_learned_rewards.append(avg_learned_reward)

            num_target_reached = data["num_target_reached"]
            avg_num_reach = np.sum(num_target_reached)/len(num_target_reached)
            avg_num_reaches.append(avg_num_reach)
            
            num_env_success = np.count_nonzero(num_target_reached==1)
            num_env_success = num_env_success / len(num_target_reached)
            percent_num_env_success.append(num_env_success)

        episodes = [i for i in range(0, num_episodes, step)]
        plot_curve(xs=episodes, ys=avg_gt_rewards, title="average gt rewards of DvrkPush", path=plot_path, x_label="episodes", y_label="avg reward")
        plot_curve(xs=episodes, ys=avg_learned_rewards, title="average learned rewards of DvrkPush", path=plot_path, x_label="episodes", y_label="avg reward")
        plot_curve(xs=episodes, ys=avg_num_reaches, title="average target reached of DVrkPush", path=plot_path, x_label="episodes", y_label="avg number of target reached")
        plot_curve(xs=episodes, ys=percent_num_env_success, title=f"fraction of envs reaching target in DvrkPush", path=plot_path, x_label="episodes", y_label="fraction of env reaching the target")

    if record_video:
        ############ images in the episode for 1 env
        data = get_episode_data(data_recording_path, episode_idx=last_episode_idx)
        images = data["images"]
        images_env_0 = [im[0, :, :, :] for im in images]
        print(f"num images in one episode: {len(images_env_0)}")

        for env_idx, im in enumerate(images_env_0):
            processed_image = Image.fromarray(im)
            processed_image.save(os.path.join(video_path, f"img_{env_idx}.png"))

        ################## track num balls reached per step in one episode (avg and in env 0)
        num_target_reached_per_step = data["target_reached_per_step"]
        avg_num_target_reached_per_step = [np.sum(num_target_reached_per_step[i])/len(num_target_reached_per_step[i]) for i in range(len(num_target_reached_per_step))]
        env_0_num_target_reached_per_step = [num_target_reached_per_step[i][0] for i in range(len(num_target_reached_per_step))]
        timesteps = [i for i in range(len(num_target_reached_per_step))]

        eef_box_dists = data["eef_box_dist"]
        avg_eef_box_dists = [np.sum(eef_box_dists[i])/len(eef_box_dists[i]) for i in range(len(eef_box_dists))]
        env_0_eef_box_dists = [eef_box_dists[i][0] for i in range(len(eef_box_dists))]
        plot_curve(xs=timesteps, ys=avg_num_target_reached_per_step, title=f"average num of target reached per step in episode {last_episode_idx}", path=plot_path, x_label="timestep", y_label="avg number of target reached")
        plot_curve(xs=timesteps, ys=env_0_num_target_reached_per_step, title=f"env 0 num of target reached per step in episode {last_episode_idx}", path=plot_path, x_label="timestep", y_label="number of target reached")
        plot_curve(xs=timesteps, ys=avg_eef_box_dists, title=f"average distance between eef and box in episode {last_episode_idx}", path=plot_path, x_label="timestep", y_label="distance")
        plot_curve(xs=timesteps, ys=env_0_eef_box_dists, title=f"env 0 distance between eef and box in episode {last_episode_idx}", path=plot_path, x_label="timestep", y_label="distance")



        ######### make video ##########
        print("making video from images...")
        image_filenames = [f"img_{i}.png" for i in range(len(images_env_0))]
        make_video(video_path, image_filenames, save_path=f'{root_folder}/{reward_folder}/rl_video_{last_episode_idx}.mp4' , fps=2)





    