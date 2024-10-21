from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np

corr = {
    "Ignore":{"input":0, "output":(255, 255, 255)},
    "Background":{"input":1, "output":(0, 0, 0)},
    "BorneIncendie":{"input":2, "output":(255, 0, 0)}
}

class BMLQ_Dataset(Dataset):
    def __init__(self, img_folders, ann_folders, transform=None):
        self.img_folders = img_folders
        self.ann_folders = ann_folders
        self.img_files = sorted(os.listdir(img_folders))
        self.ann_files = sorted(os.listdir(ann_folders))
        self.transform = transform


    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, idx):

        #img = cv.cvtColor(cv.imread(os.path.join(self.img_folders, self.img_files[idx]), cv.IMREAD_UNCHANGED), cv.COLOR_BGR2RGB)
        #ann = cv.imread(os.path.join(self.ann_folders, self.ann_files[idx]), cv.IMREAD_UNCHANGED)

        img = np.array(Image.open(os.path.join(self.img_folders, self.img_files[idx])))
        ann = np.array(Image.open(os.path.join(self.ann_folders, self.ann_files[idx])))
        anns = np.ndarray((len(corr), ann.shape[0], ann.shape[1]), dtype=np.uint8)

        j = 0
        for i in corr:
            anns[j] = (ann == corr[i]["input"])*1
            j+=1
        if self.transform is not None:
            img,anns = self.transform(img, anns)
        return img, anns
