import numpy as np
import random

def is_overlap(p1, p2, max_dist=0.005):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2 <=max_dist

np.random.seed(2021)
random.seed(2021)

init_pose = [0.0, -0.5, 0.022]
box_pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3))
cone_pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3))
print("box pose: ", box_pose)
print("cone pose: ", cone_pose)

while is_overlap(cone_pose, box_pose):
    cone_pose = list(np.array(init_pose) + np.random.uniform(low=[-0.1,-0.05,0], high=[0.1,0.05,0], size=3))
    print("new cone pose: ", cone_pose)









