Reward Learning from Suboptimal Demonstrations with Applications in Surgical Electrocautery (ISMR 2024)
====================

# Installation and Documentation
## Conda environments ready to use:
* rlgpu.yml: for collecting data in Isaac Gym Simulation and training models
* rlgpu2.yml: for RL
* emd_7: for training autoencoder using earth mover distance
* py39: miscellaneous procedures such as creating meshes and urdf

* Install Isaac Gym: Carefully follow the official installation guide and documentation from Isaac Gym. You should be able to find the documentation on isaacgym/docs/index.html. Make sure you select NVIDIA GPU before running the examples.
* Set up:
```sh
# Step 1: Create a catkin workspace called catkin_ws: http://wiki.ros.org/catkin/Tutorials/create_a_workspace

# Step 2: Clone this repo into the src folder
cd src
git clone <github link of this repo>

# Step 3: Clone other repos into the src folder:
git clone https://github.com/eric-wieser/ros_numpy.git
git clone https://github.com/gt-ros-pkg/hrl-kdl.git
git clone https://baotruyenthach@bitbucket.org/robot-learning/point_cloud_segmentation.git
```
# Collect demonstrations, Obtain preference rankings, Reward learning, Visualization
In the teleoperation folder, there are different task folders. The traceSurface folder is for the Sphere Task and the cuttingTask folder is for the Cutting Task. Each contains run.sh script that has examples of how to run the code to collect demonstrations, process data, train models, visualize results. 

## to run the motion planner for collecting demonstration.
```
roslaunch shape_servo_control dvrk_isaac.launch
```
Remember to source the workspace (source ~/catkin_ws/devel/setup.bash) before running any ros code. 


# Run RL in Isaac Gym
* RL code is in the rlgpu folder
* I highly recommend that you follow the RL examples and explanations in the official documentation before running my code. Also briefly look through the example codes. They are simpler and easier to read while having the same structure as mine.
* If you want to create a new RL task, the official documentation has a section about how to do that.
* TraceCurve and Cut are the two tasks I defined for the paper.

```
example: python train.py --task TraceCurve --logdir logs\traceCurve_learned_reward
```

# List of ROS Packages:
* shape_servo_control **[maintained]**
  * Main files to launch simulations of the dVRK manipulating deformable objects
  * util files to create deformable objects' .tet mesh files and URDF files, which can then be used to launch the object into the simulation environment.
* dvrk_description **[maintained]**
  * URDF files & CAD models & meshes
* dvrk_moveit_config **[maintained]**
  * Moveit config files. Used to control dVRK motion.

