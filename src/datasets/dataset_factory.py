from os import path
from torchvision import datasets


class DatasetFactory:
    def __init__(self, org_data_dir="../../data",):
        self.org_data_dir = org_data_dir

    def get_dataset(self, mode, dataset_name, transformer=None):

        if mode not in ["train", "test", "val"]:
            print("[ Dataset Mode should either be train/test/val ]")
            exit()
        else:
            dataset_dir = path.join(self.org_data_dir, dataset_name)
            dataset = None

            if dataset_name == "mnist":
                print("[ Dataset : mnist ]")
                dataset = (datasets.MNIST(
                    self.org_data_dir, train=True, download=True, transformer=transformer), datasets.MNIST(
                    self.org_data_dir, train=False, download=True, transformer=transformer))
            else:
                print("[ Dataset not found ]")
                exit()

        return dataset
