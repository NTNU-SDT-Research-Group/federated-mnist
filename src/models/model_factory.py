import torch

from .cnn_mnist import CNNMnist

class ModelFactory():
    def __init__(self):
        pass

    def get_model(self, model_name, num_channels=1, num_classes=1):
        model = None

        if model_name == 'cnn-mnist':
            print('[ Model : cnn-mnist ]')
            model = CNNMnist(num_channels, num_classes)

        return model
