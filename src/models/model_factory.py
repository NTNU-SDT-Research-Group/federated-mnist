import torch

from .cnn_mnist import CNNMnist

class ModelFactory():
    def __init__(self):
        pass

    def get_model(self, model_name, num_channels=1, num_classes=1, pred_type="regression"):
        if pred_type == 'regression':
            adjusted_num_classes = 1
        elif pred_type == 'mixed':
            adjusted_num_classes = num_classes + 1
        else:
            adjusted_num_classes = num_classes

        model = None

        if model_name == 'cnn-mnist':
            model = CNNMnist(num_channels, adjusted_num_classes)

        return model
