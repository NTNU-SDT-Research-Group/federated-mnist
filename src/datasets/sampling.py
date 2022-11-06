import numpy as np

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_custom_non_iid(dataset):
    print("[Custom non-iid]")
    """
    We need 10 users for this custom case
    """
    dict_users = {
        0: set(),
        1: set(),
        2: set(),
        3: set(),
        4: set(),
        5: set(),
        6: set(),
        7: set(),
        8: set(),
        9: set()
    }
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        dict_users[label].add(idx)
    return dict_users