from os import path
from torchvision import datasets

from .sampling import mnist_iid


class DatasetFactory:
    def __init__(self, org_data_dir="./data/",):
        self.org_data_dir = org_data_dir

    def get_dataset(self, dataset_name, transformer=None, num_users=100):
        dataset_dir = path.join(self.org_data_dir, dataset_name)
        train_dataset = None
        test_dataset = None
        user_groups = None

        if dataset_name == "mnist":
            print("[ Dataset : mnist ]")
            train_dataset = datasets.MNIST(
                dataset_dir, train=True, download=True, transform=transformer)
            
            test_dataset = datasets.MNIST(
                dataset_dir, train=False, download=True, transform=transformer)

            # We will do a IID sampling of the dataset
            user_groups = mnist_iid(train_dataset, num_users)

        return (train_dataset, test_dataset, user_groups)
