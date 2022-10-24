from torchvision.transforms import Compose, Normalize, ToTensor

class TransformerFactory:
    def __init__(self):
        pass

    def get_transformer(self, pipe_type="default"):
        if pipe_type == "default":
            return Compose(
                [
                    ToTensor(),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
