from .base_options import BaseOptions
import yaml

class TestOptions(BaseOptions):
    
    def __init__(self, config_file):
        BaseOptions.__init__(self, config_file)
        self.initialize_test()
    
    def initialize_test(self):
        with open(self.filename) as file:
            config = yaml.safe_load(file)
        
        self.results_dir = config['evaluation']['results_dir']
        self.ytest = config['dataset']['ytest']
        self.xtest = config['dataset']['xtest']
        self.num_test = config['evaluation']['num_test']
        self.phase = 'test'
        self.isTrain = False
        
        
        