from torch.optim import (SGD)


class OptimiserFactory:
    def __init__(self):
        pass

    def get_optimiser(self, params, optimiser_name, hyper_params):
        if optimiser_name == 'sgd':
            print("[ Optimiser : SGD ]")
            return SGD(params, lr=hyper_params['learning_rate'], momentum=hyper_params['momentum'])
