from torch.nn import (CrossEntropyLoss, NLLLoss, MSELoss)
from .utils import (RegressionLossWrapper)


class LossFactory:
    def __init__(self):
        pass

    def get_pure(self, function_name, hyper_params=None):
        loss_function = None

        if function_name == 'negative-log-likelihood-loss':
            print("[ Loss : Negative Log Likelihood Loss ]")
            loss_function = NLLLoss()

        return loss_function

    def get_loss_function(self, function_name, pred_type = "regression", hyper_params=None):
        wrapped_loss_function = None

        if pred_type == 'regression':
            wrapped_loss_function = RegressionLossWrapper(
                self.get_pure(function_name, hyper_params))

        return wrapped_loss_function