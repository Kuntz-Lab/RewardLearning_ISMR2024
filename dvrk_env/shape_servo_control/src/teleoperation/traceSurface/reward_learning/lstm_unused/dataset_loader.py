import torch
import os
from torch.utils.data import Dataset
import pickle


class TrajDataset(Dataset):

    def __init__(self, dataset_path):
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
        data = self.load_pickle_data(self.filenames[idx])

        obj_traj_i, obj_traj_j = data["emb_traj_1"], data["emb_traj_2"]
        eef_traj_i, eef_traj_j = data["eef_traj_1"], data["eef_traj_2"]
        label = data["label"]
        
        sample = (obj_traj_i, obj_traj_j, eef_traj_i, eef_traj_j, label)  
        #sample = eef_traj_i

        
        return sample          



        