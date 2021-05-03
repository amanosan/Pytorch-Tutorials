import albumentations as A
import numpy as np
from PIL import Image
from utils import plot_examples
from albumentations import augmentations
import cv2

# we can have multiple masks too
image = Image.open('./images/elon.jpeg')
mask = Image.open('./images/mask.jpeg')
mask2 = Image.open('./images/second_mask.jpeg')

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
mask2 = np.array(mask2)
for i in range(5):
    augmentations = my_transforms(image=image, masks=[mask, mask2])
    augmented_img = augmentations["image"]
    augmented_mask = augmentations["masks"]
    img_list.append(augmented_img)
    img_list.append(augmented_mask[0])
    img_list.append(augmented_mask[1])

# plotting the images and masks
plot_examples(img_list)
