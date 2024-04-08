import torch
import os
import numpy as np
import ast
import random
from torch.utils.data import Dataset
import pickle
import open3d
import sklearn
from farthest_point_sampling import *

                       



class AEDataset(Dataset):
    """predict mani point using segmentation"""


    def __init__(self, dataset_path):
        """
        Args:

        """ 

        # self.dataset_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/processed_seg_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/manipulation_points/bimanual/multi_boxes_1000Pa/processed_seg_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa_2/processed_seg_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/teleoperation/sanity_check_examples/ex_2/autoencoder/data"
        self.dataset_path = dataset_path

        self.filenames = os.listdir(self.dataset_path)
        

 


    def load_pickle_data(self, filename):
        if os.path.getsize(os.path.join(self.dataset_path, filename)) == 0: 
            print(filename)
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):        
        sample = self.load_pickle_data(self.filenames[idx])

        # pc = torch.tensor(sample["full_pc"]).float()  # shape (B, 3, num_pts)  
        pc = torch.tensor(sample["partial_pc"]).permute(1,0).float()  # shape (B, 3, num_pts)  
        #print(pc.shape)
        
        sample = pc  

        
        return sample          



        