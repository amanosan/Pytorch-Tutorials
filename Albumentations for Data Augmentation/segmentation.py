import albumentations as A
import numpy as np
from PIL import Image
from utils import plot_examples
from albumentations import augmentations
import cv2

image = Image.open('./images/elon.jpeg')
mask = Image.open('./images/mask.jpeg')

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

img_list = [image]
image = np.array(image)
mask = np.array(mask)
for i in range(5):
    augmentations = my_transforms(image=image, mask=mask)
    augmented_img = augmentations["image"]
    augmented_mask = augmentations["mask"]
    img_list.append(augmented_img)
    img_list.append(augmented_mask)

# plotting the images and masks
plot_examples(img_list)
