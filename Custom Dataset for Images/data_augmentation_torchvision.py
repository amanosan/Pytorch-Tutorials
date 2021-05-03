from torchvision.transforms.transforms import ColorJitter, RandomCrop, RandomGrayscale, RandomRotation, RandomVerticalFlip, Resize
from custom_dataset_for_images import CatsAndDogs
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch

# setting up transforms:
my_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        # resizes the image to size provided
        transforms.Resize((256, 256)),
        # crops the image randomly
        transforms.RandomCrop((224, 224)),
        # rotates the Image
        transforms.RandomRotation(degrees=45),
        # flips the image vertically
        transforms.RandomVerticalFlip(p=0.5),
        # flips the image horizontally
        transforms.RandomHorizontalFlip(p=0.5),
        # changes brightness, hue, saturation etc.
        transforms.ColorJitter(brightness=0.5),
        # conerts the image to Grayscale
        transforms.RandomGrayscale(p=0.2),
        # converts numpy array to tensor
        transforms.ToTensor(),
        # normalizing the data after converting to tensor
        # we need to find the mean and std of images in the dataset for each channel(rgb)
        # this does the following -> (value - mean) / std for each channel, thus normalizing the images
        # values 0 and 1 do nothing
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.1, 1.1, 1.1])
    ]
)

# loading the data
train_dataset = CatsAndDogs(csv_file='cats_dogs.csv',
                            root_dir='cats_dogs_resized', transform=my_transforms)

# applying transformations
img_num = 0
for (img, label) in train_dataset:
    save_image(img, f"./cats_dogs_resized/img_transformed_{img_num}.png")
    img_num += 1
