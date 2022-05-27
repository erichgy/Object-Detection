import os
import random

import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import dataprocess
import proInfor


class CustomHardNegativeDataset(Dataset):

    def __init__(self, negative_list, images, transform=None):
        self.negative_list = negative_list
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        target = 0
        negative_dict = self.negative_list[index]
        xmin,ymin,xmax,ymax = negative_dict[0:-1]
        image_id = negative_dict[-1]

        image = self.images[image_id][ymin:ymax , xmin:xmax]
        if self.transform:
            image = self.transform(image)
        return image,target,negative_dict

    def __len__(self):
        return len(self.negative_list)
