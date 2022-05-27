import numpy as np
import os
import cv2

import dataprocess
import proInfor
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, kind = None , transform=None):

        if kind is "train":
            Images_dir = proInfor.car_Images_dir
            Annotation_dir = proInfor.car_Annotations_dir
            index_dir = "./data/car_images_indexs.csv"
        else:
            Images_dir = proInfor.Vcar_Images_dir
            Annotation_dir = proInfor.Vcar_Annotations_dir
            index_dir = "./data/Tcar_images_indexs.csv"

        indexs = dataprocess.read_car_index(index_dir)

        imgs = [cv2.imread(os.path.join(Images_dir, str(index) + '.jpg'), cv2.IMREAD_COLOR) for index in
                indexs]
        positive_ann_dirs = [os.path.join(Annotation_dir, str(index) + '_1.csv') for index in indexs]
        negative_ann_dirs = [os.path.join(Annotation_dir, str(index) + '_0.csv') for index in indexs]

        positive_sizes = list()
        negative_sizes = list()

        positive_rects = list()
        negative_rects = list()

        bndbox_num = 0
        img_id = 0
        for annotations_dir in positive_ann_dirs:
            rects = np.loadtxt(annotations_dir, dtype=np.int, delimiter=',', encoding='utf-8-sig')
            bndbox_num += rects.shape[0]
            lrc = rects.tolist()
            if type(lrc[0]) == int:
                lrc.append(img_id)
            else:
                for l in lrc:
                    l.append(img_id)
            img_id += 1
            # print(lrc)
            if type(lrc[0]) == int:
                positive_rects.append(lrc)
            else:
                positive_rects.extend(lrc)
            positive_sizes.append(rects.shape[0])
        # print(postive_rects)
        #self.positive_num = bndbox_num
        bndbox_num = 0
        img_id = 0
        for annotations_dir in negative_ann_dirs:
            rects = np.loadtxt(annotations_dir, dtype=np.int, delimiter=',', encoding='utf-8-sig')
            bndbox_num += rects.shape[0]
            lrc = rects.tolist()
            if type(lrc[0]) == int:
                lrc.append(img_id)
            else:
                for l in lrc[:]:
                    l.append(img_id)
            img_id += 1
            if type(lrc[0]) == int:
                negative_rects.append(lrc)
            else:
                negative_rects.extend(lrc)
            negative_rects.extend(lrc)
            negative_sizes.append(rects.shape[0])
        self.positive_num = len(positive_rects)
        self.negative_num = len(negative_rects)
        self.transform = transform
        self.positive_rects = positive_rects
        self.negative_rects = negative_rects
        self.imgs = imgs

    def __getitem__(self, index: int):

        if index < self.positive_num :
            # print(index)
            # print("positive_num: ",len(self.positive_rects))
            rect = self.positive_rects[index]
            # print(rect)
            xmin, ymin, xmax, ymax, img_id = rect
            img = self.imgs[img_id][ymin:ymax, xmin:xmax]
            target = 1
        else:
            rect = self.negative_rects[index - self.positive_num ]
            # print(rect)
            xmin, ymin, xmax, ymax, img_id = rect
            img = self.imgs[img_id][ymin:ymax, xmin:xmax]
            target = 0
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.positive_num + self.negative_num

    def get_postive_num(self) -> int:
        return self.positive_num

    def get_negative_num(self) -> int:
        return self.negative_num
