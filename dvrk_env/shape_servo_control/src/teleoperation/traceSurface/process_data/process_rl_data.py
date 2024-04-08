
import os
import numpy as np
import pickle
import timeit
import matplotlib.pyplot as plt
from PIL import Image

import os
import moviepy.video.io.ImageSequenceClip

'''
plot data across episodes and make video for the last episode of training, also plot actions for debugging purpose
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

def plot_curves(xs, ys_list, title, path, x_label, y_label, curve_labels):
    fig, ax = plt.subplots()
    assert(len(curve_labels)==len(ys_list))
    if len(ys_list)==2:
        ax.plot(xs, ys_list[0], label=curve_labels[0])
        ax.plot(xs, ys_list[1], label=curve_labels[1], linestyle='dashed')
    else:
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
    max_num_balls = 2
    root_folder = "/home/dvrk/RL_data/traceCurve"
    reward_folder = "learned_reward_dof_vel_try" #"learned_reward_eef_pos_2ball"
    data_recording_path = f"{root_folder}/{reward_folder}/data"
    num_episodes = 701 #621 #len(os.listdir(data_recording_path)) #641 #
    last_episode_idx = num_episodes - 1
    step = 10
    print(f"num episodes: {num_episodes}")
    video_path = f"{root_folder}/{reward_folder}/videos/videos_{last_episode_idx}"
    plot_path = f"{root_folder}/{reward_folder}"
    os.makedirs(video_path, exist_ok=True)

    plot_data = True
    record_video = True
    plot_vel_actions = False
    plot_pos_actions = False
    ######### plot rewards and success rate #############
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

            num_balls_reached = data["num_balls_reached"]
            avg_num_reach = np.sum(num_balls_reached)/len(num_balls_reached)
            avg_num_reaches.append(avg_num_reach)
            
            num_env_success = np.count_nonzero(num_balls_reached==max_num_balls)
            num_env_success = num_env_success / len(num_balls_reached)
            percent_num_env_success.append(num_env_success)

        episodes = [i for i in range(0, num_episodes, step)]
        plot_curve(xs=episodes, ys=avg_gt_rewards, title="average gt rewards of TraceCurve", path=plot_path, x_label="episodes", y_label="avg reward")
        plot_curve(xs=episodes, ys=avg_learned_rewards, title="average learned rewards of TraceCurve", path=plot_path, x_label="episodes", y_label="avg reward")
        plot_curve(xs=episodes, ys=avg_num_reaches, title="average num balls reached of TraceCurve", path=plot_path, x_label="episodes", y_label="avg number of balls reached")
        plot_curve(xs=episodes, ys=percent_num_env_success, title=f"fraction of envs reaching {max_num_balls} balls in TraceCurve", path=plot_path, x_label="episodes", y_label="fraction of env reaching two balls")


    ################## make video and plot actions in the last episode ############################
    data = get_episode_data(data_recording_path, episode_idx=last_episode_idx)

    #print(data.keys())
    if record_video:
        ############ images in the episode for 1 env
        images = data["images"]
        images_env_0 = [im[0, :, :, :] for im in images]
        print(f"num images in one episode: {len(images_env_0)}")

        for env_idx, im in enumerate(images_env_0):
            processed_image = Image.fromarray(im)
            processed_image.save(os.path.join(video_path, f"img_{env_idx}.png"))

        ################## track num balls reached per step in one episode (avg and in env 0)
        balls_reached_per_step = data["balls_reached_per_step"]
        avg_balls_reached_per_step = [np.sum(balls_reached_per_step[i])/len(balls_reached_per_step[i]) for i in range(len(balls_reached_per_step))]
        env_0_balls_reached_per_step = [balls_reached_per_step[i][0] for i in range(len(balls_reached_per_step))]
        timesteps = [i for i in range(len(balls_reached_per_step))]
        plot_curve(xs=timesteps, ys=avg_balls_reached_per_step, title=f"average num balls reached per step in episode {last_episode_idx}", path=plot_path, x_label="timestep", y_label="avg number of balls reached")
        plot_curve(xs=timesteps, ys=env_0_balls_reached_per_step, title=f"env 0 num balls reached per step in episode {last_episode_idx}", path=plot_path, x_label="timestep", y_label="number of balls reached")

        ######### make video ##########
        print("making video from images...")
        image_filenames = [f"img_{i}.png" for i in range(len(images_env_0))]
        make_video(video_path, image_filenames, save_path=f'{root_folder}/{reward_folder}/rl_video_{last_episode_idx}.mp4' , fps=2)


    if plot_vel_actions:
        ################## plot eef vel per timestep in each episode
        expected_eef_vels = data["expected_eef_vels"]
        actual_eef_vels = data["actual_eef_vels"]
        actual_eef_vels.pop(-1)
        # print(len(expected_eef_vels))
        # print(len(actual_eef_vels))
        assert(len(expected_eef_vels)==len(actual_eef_vels))
        if len(actual_eef_vels)!=0:
            timesteps = [i for i in range(len(actual_eef_vels))]
            print("num eef vels: ", len(timesteps))
            num_envs = actual_eef_vels[0].shape[0]
            num_envs = 1
            plot_path = os.path.join(f"{root_folder}/{reward_folder}", f"episode {last_episode_idx} actions", "eef_vel")
            os.makedirs(plot_path, exist_ok=True)

            for i in range(num_envs):
                env_eef_vels_actual = [actual_eef_vels[t][i][0:3] for t in timesteps]
                x_eef_vels_actual = [eef_vel[0] for eef_vel in env_eef_vels_actual]
                y_eef_vels_actual = [eef_vel[1] for eef_vel in env_eef_vels_actual]
                z_eef_vels_actual = [eef_vel[2] for eef_vel in env_eef_vels_actual]

                # expected velocity is in robot frame, so we multiply the x and y components by -1
                env_eef_vels_expected = [expected_eef_vels[t][i][0:3] for t in timesteps]
                x_eef_vels_expected = [-eef_vel[0] for eef_vel in env_eef_vels_expected]
                y_eef_vels_expected = [-eef_vel[1] for eef_vel in env_eef_vels_expected]
                z_eef_vels_expected = [eef_vel[2] for eef_vel in env_eef_vels_expected]

                plot_curves(xs=timesteps, ys_list=[x_eef_vels_actual, x_eef_vels_expected], title=f"{reward_folder} eef vels x in env {i}", \
                            x_label="timestep", y_label="value of eef vel in component x", curve_labels=["actual", "expected"], path=plot_path)
                plot_curves(xs=timesteps, ys_list=[y_eef_vels_actual, y_eef_vels_expected], title=f"{reward_folder} eef vels y in env {i}", \
                            x_label="timestep", y_label="value of eef vel in component y", curve_labels=["actual", "expected"], path=plot_path)
                plot_curves(xs=timesteps, ys_list=[z_eef_vels_actual, z_eef_vels_expected], title=f"{reward_folder} eef vels z in env {i}", \
                            x_label="timestep", y_label="value of eef vel in component z", curve_labels=["actual", "expected"], path=plot_path)
                
    if plot_pos_actions:
        ################## plot eef pos per timestep in each episode
        expected_eef_poses = data["expected_eef_poses"]
        actual_eef_poses = data["actual_eef_poses"]
        assert(len(expected_eef_poses)==len(actual_eef_poses))
        if len(actual_eef_poses)!=0:
            timesteps = [i for i in range(len(actual_eef_poses))]
            print("num eef vels: ", len(timesteps))
            num_envs = actual_eef_poses[0].shape[0]
            num_envs = 1
            plot_path = os.path.join(f"{root_folder}/{reward_folder}", f"episode {last_episode_idx} actions", "eef_pos")
            os.makedirs(plot_path, exist_ok=True)

            for i in range(num_envs):
                env_eef_poses_actual = [actual_eef_poses[t][i][0:3] for t in timesteps]
                x_eef_poses_actual = [eef_pos[0] for eef_pos in env_eef_poses_actual]
                y_eef_poses_actual = [eef_pos[1] for eef_pos in env_eef_poses_actual]
                z_eef_poses_actual = [eef_pos[2] for eef_pos in env_eef_poses_actual]

                env_eef_poses_expected = [expected_eef_poses[t][i][0:3] for t in timesteps]
                x_eef_poses_expected = [eef_pos[0] for eef_pos in env_eef_poses_expected]
                y_eef_poses_expected = [eef_pos[1] for eef_pos in env_eef_poses_expected]
                z_eef_poses_expected = [eef_pos[2] for eef_pos in env_eef_poses_expected]

                plot_curves(xs=timesteps, ys_list=[x_eef_poses_actual, x_eef_poses_expected], title=f"{reward_folder} eef poses x in env {i}", \
                            x_label="timestep", y_label="value of eef pos in component x", curve_labels=["actual", "expected"], path=plot_path)
                plot_curves(xs=timesteps, ys_list=[y_eef_poses_actual, y_eef_poses_expected], title=f"{reward_folder} eef poses y in env {i}", \
                            x_label="timestep", y_label="value of eef pos in component y", curve_labels=["actual", "expected"], path=plot_path)
                plot_curves(xs=timesteps, ys_list=[z_eef_poses_actual, z_eef_poses_expected], title=f"{reward_folder} eef poses z in env {i}", \
                            x_label="timestep", y_label="value of eef pos in component z", curve_labels=["actual", "expected"], path=plot_path)
                