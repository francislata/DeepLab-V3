from torchvision.datasets import Cityscapes
import torchvision.transforms as transforms
import PIL
import torch
import numpy as np

# Constants
CROP_IMG_SIZE = (768, 768)
IGNORE_IDX = 255
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

class CustomCityscapes(Cityscapes):
    """This subclass overrides the current implementation of the Cityscapes datasets by PyTorch."""
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        target = self._convert_id_to_train_id(target)

        return image, target

    def convert_train_id_to_id(self, img):
        """Converts an images using train ID to ID"""
        for label in self.classes:
            img[img == label.train_id] = label.id

        return img

    def _convert_id_to_train_id(self, target):
        """Converts each ID of a target image to train ID"""
        for label in self.classes:
            target[target == label.id] = label.train_id
            
        return target

def load_cityscapes_datasets(filepath):
    """Loads the Cityscapes datasets"""
    train_img_transforms = transforms.Compose([
        transforms.RandomResizedCrop(CROP_IMG_SIZE, scale=(0.5, 2.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
    ])
    train_anns_transforms = transforms.Compose([
        transforms.RandomResizedCrop(CROP_IMG_SIZE, scale=(0.5, 2.0), interpolation=PIL.Image.NEAREST),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda img: torch.Tensor(np.array(img)))
    ])

    eval_img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
    ])
    eval_anns_transforms = transforms.Lambda(lambda img: torch.Tensor(np.array(img)))

    train_ds = CustomCityscapes(
        filepath, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transform=train_img_transforms, 
        target_transform=train_anns_transforms
    )

    valid_ds = CustomCityscapes(
        filepath, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transform=eval_img_transforms, 
        target_transform=eval_anns_transforms
    )
    
    return train_ds, valid_ds, IGNORE_IDX
