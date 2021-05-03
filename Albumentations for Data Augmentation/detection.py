from utils import plot_examples
from albumentations import augmentations
import cv2
import albumentations as A
import numpy as np
# from PIL import Image

image = cv2.imread('./images/cat.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bboxes = [[13, 170, 224, 410]]

# There are different types of bounding boxes
# the one used above is the Pascal_voc --> (x_min, y_min, x_max, y_max)
# Other object detections have different formats ---> YOLO, COCO

transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=35, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=35, b_shift_limit=25),
        A.OneOf([
            A.ColorJitter(p=0.5),
            A.Blur(blur_limit=3, p=0.5)
        ], p=0.1)
    ], bbox_params=A.BboxParams(format="pascal_voc", min_area=2048,
                                label_fields=[], min_visibility=0.3)
)

img_list = [image]
# we don't need to convert the image to numpy array, as we used opencv
saved_bboxes = [bboxes[0]]

for i in range(15):
    augmentations = transform(image=image, bboxes=bboxes)
    augmented_img = augmentations["image"]

    if len(augmentations['bboxes']) == 0:
        continue

    img_list.append(augmented_img)
    saved_bboxes.append(augmentations['bboxes'][0])


# plotting the results
plot_examples(img_list, bboxes=saved_bboxes)
