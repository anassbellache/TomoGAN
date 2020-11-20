import argparse
import os
import torch
import yaml

class BaseOptions():

    def __init__(self, config_file):
        self.filename = config_file
        self.initialize()
    
    def initialize(self):
        with open(self.filename) as file:
            config = yaml.safe_load(file)
        
        self.name = config['basic']['name']
        self.gpu_ids = config['basic']['gpu_ids']
        self.batch_size = int(config['model']['batch_size'])
        self.image_size = int(config['model']['image_size'])
        self.checkpoints_dir = config['basic']['checkpoints_dir']
        self.num_threads = int(config['dataset']['num_threads'])
        self.depth = int(config['model']['depth'])
        self.load_epoch = config['model']['load_epoch']
        self.verbose = config['model']['verbose']
        
    
        
