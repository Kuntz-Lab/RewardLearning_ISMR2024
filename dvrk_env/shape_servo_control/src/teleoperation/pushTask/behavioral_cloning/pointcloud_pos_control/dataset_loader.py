import torch
import os
import numpy as np
from torch.utils.data import Dataset
import pickle
                       



class BCDataset(Dataset):

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
        sample = self.load_pickle_data(self.filenames[idx])

        sample = (sample["state"], sample["action"])
        
        return sample          



        