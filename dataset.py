from torchvision.datasets import Cityscapes
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import PIL
import torch
import numpy as np
import random

# Constants
CROP_IMG_SIZE = (768, 768)
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]
HFLIP_PROB = 0.5

class CustomCityscapes(Cityscapes):
    """This subclass overrides the current implementation of the Cityscapes datasets by PyTorch."""
    def __init__(self, root, split='train', mode='fine', target_type='instance', transform=None, target_transform=None):
        super(CustomCityscapes, self).__init__(root, split=split, mode=mode, target_type=target_type, transform=transform, target_transform=target_transform)
        
        if self.split == "train":
            self.random_affine_transform = transforms.RandomAffine(0, scale=(0.5, 2.0))
            self.random_crop = transforms.RandomCrop(CROP_IMG_SIZE)
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
            ])
            self.ann_transform = transforms.Lambda(lambda img: torch.Tensor(np.array(img)))
            
    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        if self.split == "train":
            # Randomly scale the image and target
            random_scale_params = self.random_affine_transform.get_params(self.random_affine_transform.degrees, self.random_affine_transform.translate, self.random_affine_transform.scale, self.random_affine_transform.shear, image.size)
            image = F.affine(image, *random_scale_params, resample=self.random_affine_transform.resample, fillcolor=self.random_affine_transform.fillcolor)
            target = F.affine(target, *random_scale_params, resample=self.random_affine_transform.resample, fillcolor=self.random_affine_transform.fillcolor)

            # Randomly crop the image and target
            i, j, h, w = self.random_crop.get_params(image, self.random_crop.size)
            image = F.crop(image, i, j, h, w)
            target = F.crop(target, i, j, h, w)

            if random.random() <= HFLIP_PROB:
                image = F.hflip(image)
                target = F.hflip(target)

            image = self.img_transform(image)
            target = self.ann_transform(target)
            
        target = self._convert_id_to_train_id(target)

        return image, target

    def convert_train_id_to_id(self, img):
        """Converts an image from train ID to ID"""
        # Note that train IDs that are ignored in evaluation will have an ID value of 0 which works because 0 is ignored in evaluation
        final_img = torch.zeros(img.size())
        
        for label in self.classes:
            if not label.ignore_in_eval:
                final_img[img == label.train_id] = label.id
                
        return final_img

    def _convert_id_to_train_id(self, target):
        """Converts each ID of a target image to train ID"""
        for label in self.classes:
            target[target == label.id] = label.train_id
            
        return target

def load_cityscapes_datasets(filepath):
    """Loads the Cityscapes datasets"""
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
        transform=None, 
        target_transform=None
    )

    valid_ds = CustomCityscapes(
        filepath, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transform=eval_img_transforms, 
        target_transform=eval_anns_transforms
    )
    
    return train_ds, valid_ds
