#!/usr/bin/env python3
import os

# img_dir = "/home/dvrk/shape_servo_data/new_task/plane_vis/6_4"
img_dir = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_2/test_videos/"
folder_name = "demo_25"
os.chdir(os.path.join(img_dir, folder_name))


def add_frames(source_frame, num_new_frames):
    import shutil
    src = os.path.join(img_dir, folder_name, f'image{source_frame:03}.png')
    for frame in range(source_frame+1, source_frame+num_new_frames+1):
        dst = os.path.join(img_dir, folder_name, f'image{frame:03}.png')
        shutil.copy(src, dst)    

add_frames(source_frame=19, num_new_frames=4)
os.system(f"ffmpeg -framerate 4 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p video_{folder_name}.mp4")
# os.system("ffmpeg -framerate 20 -i img%04d.png -pix_fmt yuv420p output_sim_goal_oriented.mp4")
