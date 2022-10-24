from torchvision.transforms import Compose, Normalize, ToTensor


class TransformerFactory:
    def __init__(self):
        pass

    def get_transformer(self, pipe_type="default"):
        if pipe_type == "default":
            print("[ Transformer : default ]")
            return Compose(
                [
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                ]
            )
