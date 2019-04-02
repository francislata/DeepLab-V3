from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np

# Constants
IMAGE_SIZE = (513, 513)
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

class ImageSegmentationDataset(Dataset):
    """This subclass represents the image segmentation dataset to use."""

    def __init__(self, num_classes, img_filepath, anns_filepath=None, max_sample_size=None):
        self.num_classes = num_classes
        self.img_filepath = img_filepath
        self.anns_filepath = anns_filepath

        self.img_transforms = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
        ])
        self.anns_transforms = transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST)
        
        if isinstance(max_sample_size, int):
            self.img_filenames = os.listdir(self.img_filepath)[:max_sample_size]
        else:
            self.img_filenames = os.listdir(self.img_filepath)

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img = self.img_transforms(Image.open(self.img_filepath + self.img_filenames[idx]).convert("RGB"))
        annotations = np.array(self.anns_transforms(Image.open(self.anns_filepath + self.img_filenames[idx][:-3] + "png")))
        return img, annotations
