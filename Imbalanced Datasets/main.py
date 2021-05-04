# There are two ways to deal with Imbalanced Datasets
# 1. Oversampling (most used, preferred)
# 2. Class Weighting (giving higher priority to the imbalanced class)

import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn


# Lets look at the CLASS WEIGHTING method:

# we will specify the weights when we define our loss function.
# the weight parameter has two parameters since we have two classes
# the first weight = 1 is for the golden retriever
# the second weight = 50 is for the swedish elkhound
# We have 50 pictures for golden retriever and only 1 for the swedish elkhound
loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1, 50]))


# Lets look at the OVERSAMPLING method:
def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
    )

    # loading the dataset
    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    # defining the class weights
    # in this case, golden retriever will get weight = 1/50, and swedish elkhound = 1
    class_weights = []
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(1/len(files))

    # defining the weights of each sample
    sample_weights = [0] * len(dataset)

    # setting the weights of each sample according to the class to which it belongs
    for index, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[index] = class_weight

    # when doing oversampling, we make 'replacement=True'
    # creating the sampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                    replacement=True)

    # loading the data
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return loader


def main():
    loader = get_loader(root_dir='dataset', batch_size=8)
    golden = 0
    swedish = 0
    for _ in range(10):
        for data, label in loader:
            golden += torch.sum(label == 0)
            swedish += torch.sum(label == 1)

    print(f"Golden Retrievers: {golden}")
    print(f"Swedish Elkhound: {swedish}")


if __name__ == '__main__':
    main()
