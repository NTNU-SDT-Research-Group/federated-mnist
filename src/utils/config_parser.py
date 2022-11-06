from os import path
from utils.print_util import cprint
import yaml


def hydrate_config(config):
    if 'mode' not in config.keys():
        raise Exception("[ Mode not found in config ]")

    if config['mode'] == 'train':
        # type
        if 'type' not in config.keys():
            config['type'] = 'base'
            
        # basics
        if 'seed' not in config.keys():
            config['seed'] = 1
        if 'epochs' not in config.keys():
            config['epochs'] = 10
        if 'local_ep' not in config.keys():
            config['local_ep'] = 10
        if 'local_bs' not in config.keys():
            config['local_bs'] = 10
        if 'num_classes' not in config.keys():
            config['num_classes'] = 10
        if 'num_users' not in config.keys():
            config['num_users'] = 100
        if 'frac' not in config.keys():
            config['frac'] = 0.1

        # dataset (train)
        if 'dataset' not in config.keys():
            config['dataset'] = "mnist"
        if 'custom_non_iid' not in config.keys():
            config['custom_non_iid'] = False

        # model
        if 'model' not in config.keys():
            config['model'] = "cnn-mnist"
        if 'lr' not in config.keys():
            config['lr'] = 0.01
        if 'num_channels' not in config.keys():
            config['num_channels'] = 1
        if 'num_filters' not in config.keys():
            config['num_filters'] = 32
        if 'max_pool' not in config.keys():
            config['max_pool'] = True

        # optimiser
        if 'optimiser' not in config.keys():
            config['optimiser'] = "sgd"
        if 'momentum' not in config.keys():
            config['momentum'] = 0.5

        # stopping-round
        if 'stopping-round' not in config.keys():
            config['stopping-round'] = 10

    return config


def get_config_data(yml_file_name):
    name = yml_file_name.split('.')[0]

    yml_path = path.join("./configs/", yml_file_name)

    stream = open(yml_path, 'r')
    config = yaml.safe_load(stream)

    config['experiment_name'] = name

    cprint("[ Config : ", config['experiment_name'], " ]", type="info1")

    return hydrate_config(config)
