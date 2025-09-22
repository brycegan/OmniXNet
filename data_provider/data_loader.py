import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)

import torch
from torch import nn
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, data_path):
        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        pass
        

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def inverse_transform(self, data):
        pass






    





