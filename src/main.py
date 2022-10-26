from utils.check_gpu import get_training_device
from utils.config_parser import get_config_data
from baseline import train as train_baseline
from federated import train as train_federated
import argparse
import os

# stop tensorboard warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# stop W&B logs
os.environ['WANDB_SILENT'] = 'true'

# from eval import eval

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("experiment_file",
                    help="The name of the experiment config file")
args = parser.parse_args()

# Get experiment config values
if args.experiment_file is None:
    exit()
config = get_config_data(args.experiment_file)

# Get GPU / CPU device instance
device = get_training_device()

if config['mode'] == 'train':
  if config['type'] == 'base':
    train_baseline(config, device)
  else:
    train_federated(config, device)
