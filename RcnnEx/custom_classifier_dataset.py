import os
import random

import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import dataprocess
import proInfor


class CustomClassifierDataset(Dataset):

    def __init__(self, kind='train' ,transform=None):

        if kind is "train":
            Images_dir = proInfor.car_Images_dir
            Annotation_dir = proInfor.car_Annotations_dir
            index_dir = "./data/car_images_indexs.csv"
        else:
            Images_dir = proInfor.Vcar_Images_dir
            Annotation_dir = proInfor.Vcar_Annotations_dir
            index_dir = "./data/Tcar_images_indexs.csv"

        indexs = dataprocess.read_car_index(index_dir)
        images = list()
        positive_rects = list()
        negative_rects = list()

        images = [cv2.imread(os.path.join(Images_dir, str(index) + '.jpg'), cv2.IMREAD_COLOR) for index in
                indexs]
        positive_ann_dirs = [os.path.join(Annotation_dir, str(index) + '_1.csv') for index in indexs]
        negative_ann_dirs = [os.path.join(Annotation_dir, str(index) + '_0.csv') for index in indexs]

        img_id = 0
        for annotations_dir in positive_ann_dirs:
            rects = np.loadtxt(annotations_dir, dtype=np.int, delimiter=',', encoding='utf-8-sig')
            lrc = rects.tolist()
            if type(lrc[0]) == int:
                lrc.append(img_id)
            else:
                for l in lrc:
                    l.append(img_id)
            img_id += 1
            if type(lrc[0]) == int:
                positive_rects.append(lrc)
            else:
                positive_rects.extend(lrc)

        img_id = 0
        for annotations_dir in negative_ann_dirs:
            rects = np.loadtxt(annotations_dir, dtype=np.int, delimiter=',', encoding='utf-8-sig')
            lrc = rects.tolist()
            if type(lrc[0]) == int:
                lrc.append(img_id)
            else:
                for l in lrc:
                    l.append(img_id)
            img_id += 1
            if type(lrc[0]) == int:
                negative_rects.append(lrc)
            else:
                negative_rects.extend(lrc)
            negative_rects.extend(lrc)

        self.positive_list = positive_rects
        self.negative_list = negative_rects
        self.images = images
        self.positive_num = len(positive_rects)
        self.negative_num = len(negative_rects)
        self.transform = transform

    def __getitem__(self, index:int):

        if index<self.positive_num:
            target = 1
            positive_dict = self.positive_list[index]
            xmin,ymin,xmax,ymax,img_id = positive_dict
            image = self.images[img_id][ymin:ymax,xmin:xmax]
            cache_dict = positive_dict
            img = self.images[img_id]
        else:
            target = 0
            negative_dict = self.negative_list[index-self.positive_num]
            xmin, ymin, xmax, ymax, img_id = negative_dict
            image = self.images[img_id][ymin:ymax, xmin:xmax]
            cache_dict = negative_dict
            img = self.images[img_id]

        x1, y1, x2, y2 = cache_dict[0:-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 235, 255), thickness=1)
        cv2.imshow("rect", img)
        cv2.waitKey()

        if self.transform:
            image = self.transform(image)

        return image,target,cache_dict

    def __len__(self):
        return self.positive_num+self.negative_num

    def get_images(self):
        return self.images

    def get_transform(self):
        return self.transform

    def get_positive_num(self):
        return self.positive_num

    def get_negative_num(self):
        return self.negative_num

    def get_positive(self):
        return self.negative_list

    def get_negative(self):
        return self.negative_list

    def set_negative(self,negative_list):
        self.negative_list = negative_list

if __name__ == '__main__':
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    classifier_set = CustomClassifierDataset('val',transform)
    index = random.randint(1,classifier_set.get_positive_num())
    print('index:',index)
    image,target,rects = classifier_set.__getitem__(index)

    print(target)
    print(rects)