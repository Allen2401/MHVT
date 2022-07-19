import yaml
import torch
from dataset.LaneDataset import LaneDataset as dataset
class Config(object):
    def __init__(self, config_path):
        self.config = {}
        self.load(config_path)

    def load(self, path):
        with open(path, 'r') as file:
            self.config_str = file.read()
        self.config = yaml.load(self.config_str, Loader=yaml.FullLoader)

    def __repr__(self):
        return self.config_str
    def get_model_parameters(self):
        return self.config['model_parameters']

    def get_dataset(self, split):
        return dataset(**self.config['dataset'][split]['parameters'])
    def get_train_setting(self):
        return self.config['train_setting']
    # def get_model(self):
    #     name = self.config['model']['name']
    #     parameters = self.config['model']['parameters']
    #     return getattr(models, name)(**parameters)
    #
    def get_optimizer(self, model_parameters):
        return getattr(torch.optim, self.config['train_setting']['optimizer']['name'])(filter(lambda p: p.requires_grad, model_parameters),
                                                                      **self.config['train_setting']['optimizer']['parameters'])

    def get_lr_scheduler(self, optimizer):
        return getattr(torch.optim.lr_scheduler,
                       self.config['train_setting']['lr_scheduler']['name'])(optimizer, **self.config['train_setting']['lr_scheduler']['parameters'])

    def get_loss_parameters(self):
        return self.config['loss_parameters']

    def get_test_parameters(self):
        return self.config['test_parameters']

    def __getitem__(self, item):
        return self.config[item]
sys_config = Config("./config/VTLSTR_CHUAN.yaml")