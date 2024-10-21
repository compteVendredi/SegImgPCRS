import os
import lightning as L
from torch.utils.data import DataLoader
from .BMLQ_Dataset import BMLQ_Dataset
import numpy as np
import torch
from torchvision.transforms import v2
import cv2 as cv

class BMLQ_Datamodule(L.LightningDataModule):
    def __init__(self, data_dir, gaussian_ann=False, preprocess_for_resnet=False):
        self.img_dir = os.path.join(data_dir, "img_dir")
        self.ann_dir = os.path.join(data_dir, "ann_dir")
        self.gaussian_ann = gaussian_ann
        self.preprocess_for_resnet = preprocess_for_resnet

    def prepare_data(self):

        transform = v2.Compose([
            lambda img,ann:(img,ann.astype(np.float32)*255 if self.gaussian_ann else ann),
            lambda img,ann:(img, np.stack([ann[i] if i==0 else cv.GaussianBlur(ann[i], (5,5), 0) for i in range(len(ann))]) if self.gaussian_ann else ann),
            lambda img,ann:(img, ann/255 if self.gaussian_ann else ann),
            lambda img,ann:(
            (
                torch.from_numpy((np.swapaxes(np.swapaxes(img, 2, 1), 1, 0)/255).astype(np.float32))),
                torch.from_numpy(ann)
            ),
            lambda img,ann:(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img) if self.preprocess_for_resnet else img,ann),
            ])

        augmented_transform = v2.Compose([
            lambda img, ann: transform(img, ann),
            lambda img, ann: (v2.functional.horizontal_flip(img),v2.functional.horizontal_flip(ann)) if torch.randint(0, 4, (1,))[0]==0 else (img,ann),
            lambda img, ann: (v2.functional.vertical_flip(img),v2.functional.vertical_flip(ann)) if torch.randint(0, 4, (1,))[0]==0 else (img,ann),
            lambda img, ann: (lambda img, ann, degree: (v2.functional.rotate(img, degree),v2.functional.rotate(ann, degree)))(img,ann,torch.randint(0, 4, (1,))[0].float()*90)
            #lambda img, ann: v2.RandomHorizontalFlip(p=0.5),
            #v2.RandomVerticalFlip(p=0.5),
            #lambda x: v2.functional.rotate(x, torch.randint(0, 4, (1,))[0].float()*90),
            #v2.ColorJitter(contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-18/255, 18/255)),
            #v2.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
            ])

        self.train_ds = BMLQ_Dataset(os.path.join(self.img_dir, "train"), os.path.join(self.ann_dir, "train"), transform=augmented_transform)

        self.val_ds = BMLQ_Dataset(os.path.join(self.img_dir, "val"), os.path.join(self.ann_dir, "val"), transform=transform)

        self.test_ds = BMLQ_Dataset(os.path.join(self.img_dir, "test"), os.path.join(self.ann_dir, "test"), transform=transform)

    def train_dataloader(self, batch_size=8, num_workers=19):
        return DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    def val_dataloader(self, batch_size=1, num_workers=19):
        return DataLoader(self.val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self, batch_size=1, num_workers=19):
        return DataLoader(self.test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def teardown(self, stage=None) :
        pass




