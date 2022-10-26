from torch.nn import (NLLLoss)


class LossFactory:
    def __init__(self):
        pass

    def get_loss_function(self, function_name, silent=False):
        loss_function = None

        if function_name == 'negative-log-likelihood-loss':
            if not silent:
                print("[ Loss : Negative Log Likelihood Loss ]")

            loss_function = NLLLoss()

        return loss_function
