# Data Augmentation for classification task using Albumentations

from albumentations import augmentations
import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image

image = Image.open('./images/elon.jpeg')

# the transforms are applied sequentially
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
        ], p=1.0)
    ]
)

imgs_list = [image]
image = np.array(image)
for i in range(15):
    # returns a dictionary
    augmentations = my_transforms(image=image)
    # getting the image from the dictionary
    augmentated_img = augmentations["image"]
    imgs_list.append(augmentated_img)

# plotting the images (original + augmented)
plot_examples(imgs_list)
