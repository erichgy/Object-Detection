import random
import cv2
import numpy as np
import os

from numba import double

import ImageUtil
import dataprocess
import proInfor
import fileuntils as fu
import CarDataset
import matplotlib.pyplot as plt
import custom_batch_sampler
from torch.utils.data import DataLoader
import torch.nn.modules as models

import torchvision.transforms as transform
import torchvision.models as tmodel
from PIL import Image

if __name__ == '__main__':
    # car_label_dir = os.path.join(proInfor.VlabelText_dir, "car_test.txt")
    # labels = []
    # cars_index = []

    num = 3
    print(num)
    print(float(num))

    # fu.Create_folder("./data/car_images")
    # fu.Create_folder("./data/test_car_annotations")
    # with open(car_label_dir, encoding='utf-8') as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         labels = line.split()
    #         if labels[1] == "1":
    #             cars_index.append(labels[0])
    #             #fu.Copyfile(os.path.join(proInfor.testSets_dir, labels[0] + ".jpg"), proInfor.Vcar_Images_dir)
    # dataprocess.save_car_index("./data/Tcar_images_indexs.csv",cars_index)
    # dataprocess.bndbox2csv(cars_index,proInfor.Vannotations_dir,proInfor.Vcar_Annotations_dir,proInfor.car_ann)
    # car_indexs, bndboxs = dataprocess.get_bndbox()
    # img = cv2.imread('./data/car_images/' + cars_index[random.randint(0, 50)] + '.jpg', cv2.IMREAD_COLOR)
    # rects = ImageUtil.get_rects(img,'q')

    # index_dir = "./data/Tcar_images_indexs.csv"
    # cars_index = dataprocess.read_car_index(index_dir)
    # dataprocess.bndbox2csv(cars_index, proInfor.Vannotations_dir, proInfor.Vcar_Annotations_dir, proInfor.car_ann)

    # rects = np.loadtxt("./data/car_annotations/000012_1.csv",dtype=np.int,delimiter=',',encoding='utf-8-sig')
    # print(rects.tolist())
    # a = [[1,2,3,4],[2,3,4,5]]
    # print(a)
    # a.append([[3,4,5,6]])
    # print(a)
    # a.extend([[4, 5, 6, 7],[6,3,4,5]])
    # print(a)

    # plt.figure()
    # for i in range(0, 10):
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(imgs[i])
    #
    # plt.show()

    f = tmodel.inception_v3()
