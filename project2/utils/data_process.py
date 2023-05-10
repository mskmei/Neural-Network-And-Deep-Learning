import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class cutout(object):
    def __init__(self, n, length):
        self.n = n
        self.length = length

    def __call__(self, image):
        height = image.size(1)
        width = image.size(2)

        mask = np.ones((height, width), np.float32)

        for i in range(self.n):
            u = np.random.randint(height)
            v = np.random.randint(width)

            u1 = np.clip(u - self.length // 2, 0, height)
            u2 = np.clip(u + self.length // 2, 0, height)
            v1 = np.clip(v - self.length // 2, 0, width)
            v2 = np.clip(v + self.length // 2, 0, width)

            mask[u1: u2, v1: v2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image = image * mask

        return image


def get_dataloader(batch_size=16, train_part=0.8, num_workers=0):
    """
    Note: this part I partly follow some empirical knowledge online, and try to make effective data preprocessing

    batch_size: a batch of images to be loaded
    train_part: proportion of train set that will be used to train
    """
    root = './data'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #R,G,B每层的归一化用到的均值和方差
        cutout(n=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    validation_set = datasets.CIFAR10(root=root, train=True, download=True,transform=transform_test)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    train_number = len(train_set)
    index = list(range(train_number))
    np.random.shuffle(index)
    train_part_index = int(train_number*train_part)
    train_index = index[0:train_part_index]
    validation_index = index[train_part_index:]
    sampler_1 = SubsetRandomSampler(train_index)
    sampler_2 = SubsetRandomSampler(validation_index)

    train_Loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler_1, num_workers=num_workers)
    validation_Loader = DataLoader(validation_set, batch_size=batch_size, sampler=sampler_2, num_workers=num_workers)
    test_Loader = DataLoader(test_set, batch_size=batch_size,num_workers=num_workers)

    return train_Loader, validation_Loader, test_Loader
