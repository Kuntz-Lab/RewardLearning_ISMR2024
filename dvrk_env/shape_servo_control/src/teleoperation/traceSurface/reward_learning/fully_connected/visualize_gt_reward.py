import matplotlib.pyplot as plt
import numpy as np
import math

def is_overlap(p1, p2, max_dist):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2 <=max_dist

def gt_reward_function(eef_pose, balls_xyz):
    reward = 0
    for i, ball_pose in enumerate(balls_xyz):
        if ball_pose[0] == 100 and ball_pose[1] ==100 and ball_pose[2] ==-100:
            continue
        max_reward = 200
        radius = 0.00025
        for r in range(20):
            if is_overlap(eef_pose, balls_xyz[i], max_dist=radius*(2**r)):
                reward = max(reward, max_reward*(0.5**r))
    return reward

def gt_reward_function_inv_dist(eef_pose, balls_xyz, last_ball):
    reward = -math.inf
    if len(balls_xyz)==0:
        reward = 1/(np.sum((eef_pose - last_ball)**2)+1e-4)
        return reward
    for i, ball_pose in enumerate(balls_xyz):
        reward = max(1/(np.sum((eef_pose - ball_pose)**2)+1e-4), reward)
    return reward


def gt_reward_function_inv_dist_sum(eef_pose, balls_xyz, last_ball):
    reward = 0
    if len(balls_xyz)==0:
        reward = 1/(np.sum((eef_pose - last_ball)**2)+1e-2)
        return reward
    for i, ball_pose in enumerate(balls_xyz):
        reward += 1/(np.sum((eef_pose - ball_pose)**2)+1e-2)
    return reward


def gt_reward_function_neg_dist(eef_pose, balls_xyz, last_ball):
    reward = -math.inf
    scale = 1#10000
    if len(balls_xyz)==0:
        reward = -scale*np.sum((eef_pose - last_ball)**2)
        return reward
    for i, ball_pose in enumerate(balls_xyz):
        reward = max(-scale*(np.sum((eef_pose - ball_pose)**2)), reward)
    return reward


def gt_reward_function_gaussian(eef_pose, balls_xyz, last_ball):
    reward = 0
    var = 0.05
    if len(balls_xyz)==0:
        reward = 1/(var*(2*np.pi)**0.5)*np.exp(-np.sum((eef_pose - last_ball)**2)/(2*var**2))
        return reward
    reward = -math.inf
    for i, ball_pose in enumerate(balls_xyz):
        gauss_reward = 1/(var*(2*np.pi)**0.5)*np.exp(-np.sum((eef_pose - ball_pose)**2)/(2*var**2))
        reward = max(gauss_reward, reward)
    return reward

rewards = []
num_samples = 10000
eef_poses = []
balls_xyz = np.array([[-0.05, -0.45, 0.1], [-0.01, -0.51, 0.1]])
#balls_xyz = np.array([[-0.02, -0.5, 0.1]])
#balls_xyz = np.array([])

for sample in range(num_samples):
    x = np.random.uniform(low=-0.1, high=0.1)
    y = np.random.uniform(low=-0.6, high=-0.4)
    #z = np.random.uniform(low=0.011, high=0.2)
    z = 0.1
    eef_pose = np.array([x,y,z])
    eef_poses.append(eef_pose)
    rew = gt_reward_function_gaussian(eef_pose, balls_xyz, [-0.02, -0.5, 0.1])
    #rew = gt_reward_function(eef_pose, balls_xyz)
    rewards.append(rew)
    
max_reward = max(rewards)
min_reward = min(rewards)


############# 2D ####################
# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection="rectilinear")

# for ball_pose in balls_xyz:
#     ax.plot(ball_pose[0], ball_pose[1], "o", markersize=20)

# xs = [eef_poses[t][0] for t in range(num_samples)]
# ys = [eef_poses[t][1] for t in range(num_samples)]

# heats = [[(rewards[t] - min_reward) / (max_reward - min_reward), 0, 0] for t in range(num_samples)]

# ax.scatter(xs, ys, c=heats) 

# plt.title(f"GT reward over eef_samples")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# plt.show()

############# 3D function plot #####################
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

for ball_pose in balls_xyz:
    ax.plot(ball_pose[0], ball_pose[1], ball_pose[2], "o", markersize=20)

xs = [eef_poses[t][0] for t in range(num_samples)]
ys = [eef_poses[t][1] for t in range(num_samples)]
zs = [(rewards[t]) for t in range(num_samples)]
heats = [[(rewards[t] - min_reward) / (max_reward - min_reward), 0, 0] for t in range(num_samples)]
ax.scatter(xs, ys, zs, c=heats) 

plt.title(f"GT reward over eef_samples")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()