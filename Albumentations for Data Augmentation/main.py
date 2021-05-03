import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch.nn as nn
import cv2
import os
from torch.utils.data.dataset import Dataset


class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        super(ImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        root_directory = os.path.join(self.root_dir, self.class_names[label])
        img_dir = os.path.join(root_directory, img)

        # loading the image
        image = np.array(Image.open(img_dir))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return (image, label)


my_transforms = A.Compose(
    [
        A.Resize(height=1080, width=1920),
        A.RandomCrop(height=720, width=1280),
        A.Rotate(limit=45, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25),

        # applying any one of the following transformations
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5)
        ], p=1.0),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255
        ),
        ToTensorV2()
    ]
)

dataset = ImageFolder(root_dir='cats_dogs', transform=my_transforms)

# checking if the dataset is working
for (x, y) in dataset:
    print(x.shape)
    print(y)
    break
