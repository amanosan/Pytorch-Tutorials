from custom_dataset_for_images import CatsAndDogs
import torch
from torch.utils.data import DataLoader
import torchvision
import os
# Loading the data throught the class we just made 
csv_file = 'cats_dogs.csv'  # name of the csv file
root_dir = os.getcwd()
batch_size = 512

dataset = CatsAndDogs(csv_file, root_dir, transform=torchvision.transforms.ToTensor())

# train and test data:
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [8, 2])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)