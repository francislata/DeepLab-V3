from torchvision.datasets import Cityscapes
import torchvision.transforms as transforms
import PIL
import torch
import numpy as np

# Constants
IMG_SIZE = (321, 321)
IGNORE_IDX = 255
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]
CITYSCAPES_ROOT_FILEPATH = "./cityscapes-dataset"

def load_cityscapes_datasets():
    """Loads the Cityscapes datasets"""
    img_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
    ])
    anns_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=PIL.Image.NEAREST),
        transforms.Lambda(lambda img: torch.Tensor(np.array(img)))
    ])

    train_ds = Cityscapes(
        CITYSCAPES_ROOT_FILEPATH, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transform=img_transforms, 
        target_transform=anns_transforms
    )

    valid_ds = Cityscapes(
        CITYSCAPES_ROOT_FILEPATH, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transform=img_transforms, 
        target_transform=anns_transforms
    )

    return train_ds, valid_ds, IGNORE_IDX
