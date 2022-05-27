import os
import csv
import time

import proInfor
import xml.etree.ElementTree as ET
import ImageUtil
import cv2


def bndbox2csv(indexs, anntations_dir, car_ann_dir, car_ann):
    pos_num = 0
    ne_num = 0
    for index in indexs:
        since = time.time()
        xml_dir = os.path.join(anntations_dir, str(index) + ".xml")
        dom = ET.parse(xml_dir)
        root = dom.getroot()
        ann_infor = [index]

        img_path = proInfor.Vcar_Images_dir + '/' + str(index) + '.jpg'
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rects = ImageUtil.get_rects(img, 'q')

        postive_list = list()
        ne_list = list()

        for child in root.iter('object'):
            if child.find('name').text == "car" and not child.find('difficult').text == True:
                xmin = int(child.find('./bndbox/xmin').text)
                ymin = int(child.find('./bndbox/ymin').text)
                xmax = int(child.find('./bndbox/xmax').text)
                ymax = int(child.find('./bndbox/ymax').text)
                postive_list.append([xmin, ymin, xmax, ymax])

        ne_list += ImageUtil.IoU(postive_list, rects, proInfor.IOU_threshold)


        with open(os.path.join(car_ann_dir, str(index) + '_1.csv'), mode="w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(postive_list)

        with open(os.path.join(car_ann_dir, str(index) + '_0.csv'), mode="w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(ne_list)
        #ImageUtil.rect_img(img, ne_list, postive_list)
        pos_num+=len(postive_list)
        ne_num+=len(ne_list)
        print(len(ne_list))
        postive_list = []
        ne_list = []
        print("成功读取 %s !" % (str(index) + ".xml"))
        print("正向边框总数：",pos_num)
        time_elapsed = time.time() - since
        print('Cost: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        #print("负向边框总数：", ne_num)


def get_bndbox(xml_path):
    # car_anns = {}
    # car_bndboxs = []
    # ind = 0
    # with open(os.path.join(xml_path), encoding="utf-8-sig") as f:
    #     reader = csv.reader(f)
    #     ci = ""
    #     bndboxs = []
    #     for row in reader:
    #         bndboxs = []
    #         ci = list(row)[0:1]
    #         for item in list(row)[1:]:
    #             bndboxs.append(item)
    #         car_anns[str(ci)] = ind
    #         ind += 1
    #         car_bndboxs.append(bndboxs)
    # return car_anns, car_bndboxs
    positive_list = list()
    dom = ET.parse(xml_path)
    root = dom.getroot()
    for child in root.iter('object'):
        if child.find('name').text == "car" and not child.find('difficult').text == True:
            xmin = int(child.find('./bndbox/xmin').text)
            ymin = int(child.find('./bndbox/ymin').text)
            xmax = int(child.find('./bndbox/xmax').text)
            ymax = int(child.find('./bndbox/ymax').text)
            positive_list.append([xmin, ymin, xmax, ymax])

    return positive_list



def save_car_index(index_dir,indexs):
    with open(index_dir, mode='w', encoding='utf-8',newline='') as file:
        writer = csv.writer(file)
        writer.writerows(indexs)


def read_car_index(index_dir):
    car_indexs = list()
    with open(index_dir, encoding='utf-8') as file:
        reader = csv.reader(file)
        for line in reader:
            ind = ""
            for s in line:
                ind+=s
            car_indexs.append(ind)
    return car_indexs