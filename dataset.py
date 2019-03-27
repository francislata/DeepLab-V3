from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

# Constants
IMAGE_SIZE = (512, 512)
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

class ImageSegmentationDataset(Dataset):
    """This subclass represents the image segmentation dataset to use."""

    def __init__(self, img_filepath, annotations_filepath=None):
        self.img_transforms = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
        ])
        self.annotations_transforms = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])
        self.img_filepath = img_filepath
        self.annotations_filepath = annotations_filepath
        self.img_filenames = os.listdir(self.img_filepath)[:500]

        if annotations_filepath is not None:
            self.annotations_filenames = os.listdir(self.annotations_filepath)[:500]

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img = Image.open("{}{}".format(self.img_filepath, self.img_filenames[idx])).convert("RGB")
        img = self.img_transforms(img)
        annotations = self.annotations_transforms(Image.open("{}{}".format(self.annotations_filepath, self.annotations_filenames[idx]))).long()

        return img, annotations
