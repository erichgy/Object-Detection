import copy
import time

import numpy as np
import cv2
import torch
import ImageUtil
import dataprocess
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.autograd


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def get_mode(device=None):
    model = models.alexnet(pretrained=True)
    num_class = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_class)
    # model.load_state_dict(torch.load('./models/best_linear_svm_alexnet_car.pth'))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model


def draw_box_with_text(img, rect_list, score_list):
    for i in range(len(rect_list)):
        xmin, ymin, xmax, ymax = rect_list[i]
        score = score_list[i]

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=1)
        cv2.putText(img, "{:.3f}".format(score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def nms(rect_list, score_list):
    nms_rects = list()
    nms_scores = list()

    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]

    tresh = 0.3
    while len(score_array) > 0:
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])
        rect_array = rect_array[1:]
        score_array = score_array[1:]

        length = len(score_array)
        if length <= 0:
            break

        print("nms:", nms_rects[-1])
        print("rect", rect_array)
        iou_scores = ImageUtil.IoU(nms_rects[-1].tolist(), rect_array.tolist(), tresh)
        idxs = np.where(iou_scores < tresh)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

    return nms_rects, nms_scores


if __name__ == '__main__':
    device = get_device()
    model = get_mode(device)
    transform = get_transform()

    gs = ImageUtil.get_selective_search()

    test_img_path = './data/test_car_images/000014.jpg'
    test_xml_path = 'H:\data\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\Annotations/000014.xml'

    img = cv2.imread(test_img_path)
    dst = copy.deepcopy(img)

    cv2.imshow("demo", img)
    cv2.waitKey()

    bndboxs = dataprocess.get_bndbox(test_xml_path)
    print(bndboxs[0])
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)

    rects = ImageUtil.get_rects(img, 'f')
    print('候选区域：', len(rects))

    svm_thresh = 0.60

    score_list = list()
    positive_list = list()

    start = time.time()
    for rect in rects:
        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]
        rect_transform = transform(rect_img).to(device)
        output = model(rect_transform.unsqueeze(0))[0]

        if torch.argmax(output).item() == 1:
            probs = torch.softmax(output, dim=0).cpu().numpy()
            print("probs:", probs)

            if probs[1] >= svm_thresh:
                score_list.append(probs[1])
                positive_list.append(rect)
                #print(rect, output, probs)

    end = time.time()
    print('detect time:%d s' % (end - start))

    nms_rects, nms_scores = nms(positive_list, score_list)
    print(nms_rects)
    print(nms_scores)

    draw_box_with_text(dst, nms_rects, nms_scores)

    cv2.imshow('img', dst)
    cv2.waitKey(0)
