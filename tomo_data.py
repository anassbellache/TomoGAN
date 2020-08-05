import os
import sys

import h5py as h5

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import utils

class TomoData(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.archive = h5.File(data_path, 'r')
        self.data = self.archive['images']
        self.data = np.array(self.data)
        self.transform = transform
    
    def __getitem__(self, index):
        datum = self.data[index]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum
    
    def __len__(self):
        return self.data.shape[0]
    
    def close(self):
        self.archive.close()
