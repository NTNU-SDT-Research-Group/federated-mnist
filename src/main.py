import os

# stop tensorboard warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# stop W&B logs
os.environ['WANDB_SILENT'] = 'true'

import argparse
from baseline import train as train_baseline
# from eval import eval
from utils.config_parser import get_config_data
from utils.check_gpu import get_training_device